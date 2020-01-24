#include <SoftwareSerial.h>


#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <Adafruit_Sensor.h>

//proximity sensor
#define sensorPin A0

//bluetooth
SoftwareSerial bt(2, 3); // RX | TX // receive, transmit bluetooth
char ledControl = "0";
//accelerometer
Adafruit_MMA8451 mma = Adafruit_MMA8451();


//LEDS
const int greenLED = 7;
const int redLED = 6;

float gravity = -9.8;

//flag to determine which phase of the lift we are in
bool gnd;
bool up;
bool peak;

void setup()
{
  //bluetooth
  Serial.begin(9600);
  bt.begin(9600);  //Default Baud for comm, it may be different for your Module.
  Serial.println("The bluetooth gates are open.\n Connect to HC-05 from any other bluetooth device with 1234 as pairing key!.");

  //accelerometer
  if (! mma.begin()) {
    Serial.println("Couldnt start accelerometer");
    while (1);
  }
  Serial.println("MMA8451 found!");
  mma.setRange(MMA8451_RANGE_2_G);

  //LEDs
  pinMode(greenLED, OUTPUT);
  pinMode(redLED, OUTPUT);
  //flash on to indicate power
  digitalWrite(greenLED, HIGH);
  digitalWrite(redLED, HIGH);
  delay(1000);
  //default off
  digitalWrite(greenLED, LOW);
  digitalWrite(redLED, LOW);

  gnd = false;
  up = false;
  peak = false;
}

void loop()
{
  // setup accelerometer
  mma.read();
  sensors_event_t event;
  mma.getEvent(&event);
  float accel_z = event.acceleration.z;

  // setup prox sensor
  float volts = analogRead(sensorPin)*0.0048828125;
  int prox = 13*pow(volts, -1);
  
  //hold box upside down to reset and wait
  if (accel_z > 0 && prox > 30) {
    gnd = false;
    up = false;
    peak = false;
    digitalWrite(greenLED, HIGH);
    digitalWrite(redLED, HIGH);
    while (accel_z > 0) {
      mma.read();
      sensors_event_t event;
      mma.getEvent(&event);
      accel_z = event.acceleration.z;
    }
////    while (true) {
////      volts = analogRead(sensorPin)*0.0048828125;
////      prox = 13*pow(volts, -1);
////      if (prox < 30) {
////         break;
////      }
////    }
    delay(500);
    digitalWrite(greenLED, LOW);
    digitalWrite(redLED, LOW);
  }
  
  // decide what to transmit to RaspberryPi
  if (accel_z >= (gravity - 1.25) && accel_z <= (gravity + 1.25) && !gnd) {
    //gnd
    bt.write("gnd\n");
    gnd = true;
  } else if (accel_z < (gravity - 1.5) && !up && gnd) {
    //up
    bt.write("up\n");
    up = true;
  } else if (accel_z >= (gravity - 1.25) && accel_z <= (gravity + 1.25) && !peak && up && gnd) {
    //peak
    bt.write("peak\n");
    peak = true;
  }


  //LED logic
  ledControl = bt.read();
  //send 1 for "good" form, 0 for "bad" form
  if (ledControl == '1') {
    digitalWrite(greenLED, HIGH);
    digitalWrite(redLED, LOW);
    delay(5000);
    gnd = false;
    up = false;
    peak = false;
  } else if (ledControl == '0') {
    digitalWrite(greenLED, LOW);
    digitalWrite(redLED, HIGH);
    delay(5000);
    gnd = false;
    up = false;
    peak = false;
  }
  delay(250);
}

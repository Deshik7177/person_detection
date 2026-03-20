#include <ESP32Servo.h>

Servo frontLeft, frontRight, backLeft, backRight;

#define FL_PIN 13
#define FR_PIN 12
#define BL_PIN 14
#define BR_PIN 27

int stopSignal = 90;

// Your tuned values (kept exactly)
int FL_forward = 60;
int FL_backward = 120;

int FR_forward = 120;
int FR_backward = 60;

int BL_forward = 120;
int BL_backward = 60;

int BR_forward = 60;
int BR_backward = 120;

char command;

unsigned long lastCommandTime = 0;
int timeout = 300; // ms (tune this if needed)

void setup() {
  Serial.begin(115200);

  frontLeft.attach(FL_PIN);
  frontRight.attach(FR_PIN);
  backLeft.attach(BL_PIN);
  backRight.attach(BR_PIN);

  stopAll();

  Serial.println("Ready for commands...");
}

void loop() {

  // 🔹 Read command safely
  if (Serial.available()) {
    command = Serial.read();

    // ignore newline / junk
    if (command == '\n' || command == '\r') return;

    // make case insensitive
    command = toupper(command);

    lastCommandTime = millis();

    if (command == 'F') moveForward();
    else if (command == 'B') moveBackward();
    else if (command == 'L') turnLeft();
    else if (command == 'R') turnRight();
    else if (command == 'S') stopAll();
  }

  // 🔹 AUTO STOP (this is the magic)
  if (millis() - lastCommandTime > timeout) {
    stopAll();
  }
}

// Movement functions

void moveForward() {
  frontLeft.write(FL_forward);
  frontRight.write(FR_forward);
  backLeft.write(BL_forward);
  backRight.write(BR_forward);
}

void moveBackward() {
  frontLeft.write(FL_backward);
  frontRight.write(FR_backward);
  backLeft.write(BL_backward);
  backRight.write(BR_backward);
}

void turnLeft() {
  frontLeft.write(FL_backward);
  frontRight.write(FR_forward);
  backLeft.write(BL_backward);
  backRight.write(BR_forward);
}

void turnRight() {
  frontLeft.write(FL_forward);
  frontRight.write(FR_backward);
  backLeft.write(BL_forward);
  backRight.write(BR_backward);
}

void stopAll() {
  frontLeft.write(stopSignal);
  frontRight.write(stopSignal);
  backLeft.write(stopSignal);
  backRight.write(stopSignal);
}
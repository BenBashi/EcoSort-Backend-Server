#include <Servo.h> //pulse

#define servoPin 10
#define ML_Ctrl 4
#define ML_PWM 6
#define MR_Ctrl 3
#define MR_PWM 6  // kept the same PWM pin

Servo trackServo; //pulse

void setup() {
  Serial.begin(115200);
  trackServo.attach(servoPin); //pulse

  pinMode(ML_Ctrl, OUTPUT);
  pinMode(ML_PWM, OUTPUT);
  pinMode(MR_Ctrl, OUTPUT);
  pinMode(MR_PWM, OUTPUT);

  trackServo.write(0); // moves the servo to the 0 degree position AT STARTUP
  analogWrite(ML_PWM, 0);
  analogWrite(MR_PWM, 0);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "LEFT") {
      trackServo.write(0);
    } else if (cmd == "RIGHT") {
      trackServo.write(180);
    } else if (cmd == "MOTORS_FORWARD_SLOW") {
      digitalWrite(ML_Ctrl, LOW);
      analogWrite(ML_PWM, 200);
      digitalWrite(MR_Ctrl, LOW);
      analogWrite(MR_PWM, 200);
    } else if (cmd == "MOTORS_STOP") {
      analogWrite(ML_PWM, 0);
      analogWrite(MR_PWM, 0);
    } else {
      Serial.println("Unknown command!");
    }
  }
}
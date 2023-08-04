// Define the global variables
const int redLedPin = 7;
const int greenLedPin = 8;
const int motionSensorPin = 3;
const int buttonPin = 2;

volatile int pirState = LOW;
volatile int pinStatus = 0;
volatile int buttonState = 0;

void setup() {
  Serial.begin(9600);

  pinMode(redLedPin, OUTPUT);
  pinMode(greenLedPin, OUTPUT);
  pinMode(motionSensorPin, INPUT);
  pinMode(buttonPin, INPUT);

  attachInterrupt(digitalPinToInterrupt(motionSensorPin), motionDetected, CHANGE);
  attachInterrupt(digitalPinToInterrupt(buttonPin), buttonPressed, CHANGE);
}

void loop() {}

void buttonPressed()
{
  digitalWrite(greenLedPin, !digitalRead(greenLedPin));
  buttonState = !buttonState;
  Serial.println(buttonState ? "Button Pressed" : "Button Released");
}

void motionDetected()
{
  pinStatus = digitalRead(motionSensorPin);
  if (pinStatus == HIGH)  
  {            
    digitalWrite(redLedPin, HIGH);  

    if (pirState == LOW) 
  {
      Serial.println("Motion detected!");
      pirState = HIGH;
    }
  } 
  else 
  {
    digitalWrite(redLedPin, LOW);

    if (pirState == HIGH)
  {
      Serial.println("Motion no longer detected!");
      pirState = LOW;
    }
  }
}

// Define the global variables
const int ledPin = 7;
const int motionSensorPin = 2;
int pirState = LOW;
int pinStatus = 0;

void setup() {
  Serial.begin(9600);

  pinMode(ledPin, OUTPUT);
  pinMode(motionSensorPin, INPUT);

}

void loop() {
  pinStatus = digitalRead(motionSensorPin);

  Serial.print("Motion Sensor Pin is: ");
  Serial.println(pinStatus);
  
  if (pinStatus == HIGH)  
  {            
    digitalWrite(ledPin, HIGH);  

    Serial.println("LED On.");
    Serial.print("Led Pin is: ");
    Serial.println(digitalRead(ledPin));

    if (pirState == LOW) 
  {
      Serial.println("Motion detected!");
      pirState = HIGH;
    }
  } 
  else 
  {
    digitalWrite(ledPin, LOW);

    Serial.println("LED Off.");
    Serial.print("Led Pin is: ");
    Serial.println(digitalRead(ledPin));

    if (pirState == HIGH)
  {
      Serial.println("Motion no longer detected!");  // print on output change
      pirState = LOW;
    }
  }
}



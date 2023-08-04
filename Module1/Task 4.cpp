// Define the global variables
const int redLedPin = 7;
const int greenLedPin = 6;
const int motionSensorPin = 3;
const int button1Pin = 2;

const int button2Pin = 8;
const int yellowLedPin = 4;
const int blueLedPin = 5;

volatile int pinStatus = 0;

volatile int pirState = LOW;
volatile int pirPrevState = LOW;
volatile int button1State = LOW;
volatile int button1PrevState = LOW;
volatile int button2State = LOW;
volatile int button2PrevState = LOW;

volatile int blueLedState = LOW;
volatile int blueLedPrevState = LOW;

void setup() {
  Serial.begin(9600);

  pinMode(redLedPin, OUTPUT);
  pinMode(greenLedPin, OUTPUT);
  pinMode(yellowLedPin, OUTPUT);
  pinMode(blueLedPin, OUTPUT);

  pinMode(motionSensorPin, INPUT);
  pinMode(button1Pin, INPUT);
  pinMode(button2Pin, INPUT);

  attachInterrupt(digitalPinToInterrupt(motionSensorPin), motionDetected, CHANGE);
  attachInterrupt(digitalPinToInterrupt(button1Pin), buttonPressed, CHANGE);

  /* PCICR == Pin Change Interrupt Control Register
   * - Used to enable or pisable pin change interrupts.
   * - PCIE0 is used to enable Pin Change Interrupt 0 (this will change the bit at this position to 1)
   * - PCIE0 corresponds with Port B - The Digital Pins 8 - 13
   */
  PCICR |= B00000001;  //(1 << PCIE0);
  /* PCMSK0 == Pin Change Mask Register 0
   * - Used to specificy which pins in the PCINT0 set should trigger a pin change interrupt.
   * - PCINT0 corresponds with Port B0 - Arduino Digital Pin 8.
   */
  PCMSK0 |= B00000001;  //(1 << PCINT0);

  noInterrupts();  // disable all interrupts
  /* TCCR1A and TCCR1B control the mode of operations of Timer1.
   * We set these to 0 in preparation for setup of the timer.
   */
  TCCR1A = 0;
  TCCR1B = 0;

  // This resets the timer count to 0, so that it starts counting from 0 after the setup.
  TCNT1  = 0;


  
  // Turn on CTC mode by setting the WGM12 bit to 1.
  TCCR1B |= (1 << WGM12);

  // Set CS10 and CS12 bits for 1024 prescaler
  TCCR1B |= (1 << CS10);
  TCCR1B |= (1 << CS12);
  
  /* OCR1A = Output Compare Register A
   * Set the compare match value for Timer1 to 31249.
   * In CTC Mode, Timer1 will count up to this value, then reset and trigger an interrupt.
   * This should happen roughly every 2 seconds.
   * ~0.5Hz = 16000000 / (31249 * 1024)
   * 2s = (31249 + 1) * (1024/16000000)
   */
  OCR1A = 31249;

  /* Timer/Counter Interrupt Mask Register
   * Each bit in this register corresponds to different interrupt for Timer1.
   * When the OCIE1A bit is set this enables the Compare Match A interrupt for Timer1.
   */
  TIMSK1 |= (1 << OCIE1A);

  interrupts();  // enable all interrupts
}

void loop() 
{
  if (pirState != pirPrevState) {
    if (pirState == HIGH) {
      Serial.println("Motion detected!");
    } else {
      Serial.println("Motion no longer detected!");
    }
    pirPrevState = pirState;
  }

  if (button1State != button1PrevState) {
    if (button1State == HIGH) {
      Serial.println("Button_1 Pressed");
    } else {
      Serial.println("Button_1 Released");
    }
    button1PrevState = button1State;
  }

  if (button2State != button2PrevState) {
    if (button2State == HIGH) {
      Serial.println("Button_2 Pressed");
    } else {
      Serial.println("Button_2 Released");
    }
    button2PrevState = button2State;
  }

  if (blueLedState != blueLedPrevState) {
    if (blueLedState == HIGH) {
      Serial.println("Blue LED On");
    } else {
      Serial.println("Blue LED Off");
    }
    blueLedPrevState = blueLedState;
  }
}

void buttonPressed()
{
  digitalWrite(greenLedPin, !digitalRead(greenLedPin));
  button1State = !button1State;
}

void motionDetected()
{
  pinStatus = digitalRead(motionSensorPin);
  if (pinStatus == HIGH) {            
    digitalWrite(redLedPin, HIGH);  
    pirState = HIGH;
  } else {
    digitalWrite(redLedPin, LOW);
    pirState = LOW;
  }
}


/* TIMER1_COMPA_vect is the comparison interrupt vector for Timer1
 * An interrupt vector is essentially the address of the function 
 * (called an interrupt service routine, or ISR) that should 
 * be executed when a particular interrupt occurs.
 */
ISR(TIMER1_COMPA_vect) {
  digitalWrite(blueLedPin, !digitalRead(blueLedPin));  // toggle blue LED
  blueLedState = !blueLedState;
}

// PCINT0_vect is the interrupt vector for Pin Change Interrupt 0.
ISR(PCINT0_vect) {
  digitalWrite(yellowLedPin, !digitalRead(yellowLedPin));
  button2State = !button2State;
}
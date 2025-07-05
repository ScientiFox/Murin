/*
Keypad interface

A library to interface an analog signal keypad to an Arduino
*/

/*
Standard PINOUT:

PIN 1  GND
PIN 2 *
PIN 3 7
PIN 4 4
PIN 5 1
PIN 6 0
PIN 7 8
PIN 8 5
PIN 9 2
PIN 10  #
PIN 11  9
PIN 12  6
PIN 13  3
*/

//previous button press flag
String button_prev = "x";

void setup() {
  //Start the serial port
  Serial.begin(9600);
}

void loop() {

  //Read in the analog values
  int sensorValue1 = analogRead(A0);
  int sensorValue2 = analogRead(A1);
  int sensorValue3 = analogRead(A2);

  //Convert the values into button strings
  String button1 = to_button(sensorValue1,A0);
  String button2 = to_button(sensorValue2,A1);
  String button3 = to_button(sensorValue3,A2);

  //If not the initial value, update the button to the one read
  String button = "x";
  if (!button1.equals("x")){
    button = button1;}
  else if (!button2.equals("x")){
    button = button2;}
  else if (!button3.equals("x")){
    button = button3;}

  //Update button if the new reading isn't the previous one
  if (button_prev.equals("x")&!button_prev.equals(button)&!button.equals("x")){
    Serial.println(button);} //Send to serial port

  button_prev = button; //Update previous button
  delay(100); //10Hz update rate
}

//Function to convert analog values to buttons
String to_button(int val, int pin){

  //String searches for the analog read channels
  String A_0[] = {"1","4","7","*"};
  String A_1[] = {"2","5","8","0"};
  String A_2[] = {"3","6","9","#"};

  //sorting value index for channel selector
  int index = -1;

  //Sort index by read values
  if (val < 100){index = 0;}
  else if (val < 200){index = 1;}
  else if (val < 700){index = 2;}
  else if (val < 1000){index = 3;}

  //If no index, no button pressed
  if (index == -1){return "x";}
  else{ //otherwise
    if (pin == A0){return A_0[index];} //grab the index for each channel
    else if (pin == A1){return A_1[index];}
    else if (pin == A2){return A_2[index];}
  }
}


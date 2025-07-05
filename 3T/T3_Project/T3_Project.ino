/*
T3 Arduino interface

This software is the firmware to interface the serial version of the
T3 demo arduino interface to the python system over the serial port
*/

//Load in needed libraries
#include <stdint.h>
#include <TouchScreen.h> 
#include <TFT.h>

//Create image, game, and processing variables
int letterSize = 8; //Size of the letters to plot
int offset = 3.5*letterSize; //offset for letter placement
int centerX = 120; //center point of the screen
int centerY = 160;
int vShift = 79; //setp size for grid
int hShift = 74;
int i = 0; //index variable
int xp,yp; //x and y position
int moves[] = {0,0,0,0,0,0,0,0,0}; //list of moves
int Pc; //player flag

//Pressure terms for touchscreen
#define MINPRESSURE 1
#define MAXPRESSURE 1000

//Make the touchscreen object
TouchScreen ts = TouchScreen(A3, A2, A1, A0, 300);

void setup(){
  Serial.begin(9600); //Start the serial port

  Tft.init();  //init TFT library

  Tft.drawString("Loading Brain",20,20,2,RED); //Draw the loading note

  //wait for the serial start command from the Python side
  while (Serial.available() == 0){
    delay(1);} 
  Serial.read(); //Read in the data

  initBoard(); //Draw the blank board
  
  delay(1); //brief startup delay
}

//player state variable
int Pl = 0;

void loop(){
  //Make a touchscreen point
  TSPoint p = ts.getPoint();

  //Check pressure and player flag
  if ((p.z > MINPRESSURE)&(Pl == 0)) {
    xp = (p.x > 650) - (p.x < 350); //grab x and y coords
    yp = (p.y > 650) - (p.y < 420);
  
    //Check reset and cell 'clicks'
    int R = (p.x>810)*(p.y>820);
    int P = 3*(1-yp) + (1-xp);

    if (R){ //If reset button hit, clear the screen
      initBoard();}
    else if ((moves[P] == 0)&(p.x<800)*(p.x>150)*(p.y<820)*(p.y>250)){ //If in the play area
      drawMove(1,P); //Draw the player's move
      moves[P] = 1; //set moves made list
      i++; //increment counter
      for (int j = 0;j<8;j++){ //Report move to serial port
        Serial.print(moves[j]);
        Serial.print(",");}
        Serial.println(moves[8]);}
    Pl = 1; //Toggle player flag
  }

  //If there's a command from the AI player
  if (Serial.available()>0){
    Pc = Serial.parseInt(); //Parse to get the index move command
    String txt = String(Pc); //display text of the move itself
    Tft.fillRectangle(20,20,30,10,BLUE); //draw a rectangle and the mark
    Tft.drawString(&txt[0],20,20,1,RED);
    if (Pc < 9){ //If still moves left, draw the move and mark cell by player
      drawMove(0,Pc);
      moves[Pc] = -1;}  
    else if (Pc == 9){ //If a 9 move count, it's a tie
      Tft.drawString("TIE",0,40,10,RED); //note it
      delay(1000); //wait
      initBoard();}  //reset board
    else if (Pc == 10){ //If player won, note it
      Tft.drawString("WIN",0,40,10,RED);
      delay(1000);
      initBoard();}  
    else if (Pc == 11){ //if AI won, report it
      Tft.drawString("LOSE",0,40,6,RED);
      delay(1000);
      initBoard();}
    Pl = 0; //toggle player flag
  }

  delay(30); //33Hz update rate
}

//Function to draw a move by player m at spot p
void drawMove(int m, int p){
  String Move; //String for the move
  int dx = 1-p%3; //calculate grid position
  int dy = 1-p/3;
  if (m){ //for X player, put X on screen
    Tft.drawString("X",centerX-hShift*dx-offset,centerY-vShift*dy-offset,letterSize,CYAN);}
  else{ //Do same for Os
    Tft.drawString("O",centerX-hShift*dx-offset,centerY-vShift*dy-offset,letterSize,CYAN);}  
}

//Function to wipe the board
void initBoard(){
  memset(moves,0,sizeof(moves)); //Set the memory to all 0s
  i = 0; //turn index to 0
  Pl = 0; //reset player flag

  Tft.paintScreenBlack(); //Clear screen
  
  Tft.drawVerticalLine(80,55,210,WHITE); //Draw grid lines
  Tft.drawVerticalLine(160,55,210,WHITE);
  Tft.drawHorizontalLine(20,119,205,WHITE);
  Tft.drawHorizontalLine(20,198,205,WHITE);

  Tft.fillCircle(10,10,5,RED); //Draw dots
  Tft.fillCircle(10,50,5,BLUE);
  Tft.fillCircle(50,10,5,GREEN);

  }

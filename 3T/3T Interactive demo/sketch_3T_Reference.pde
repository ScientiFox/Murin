import processing.serial.*;

Serial myPort;  // The serial port:

import processing.net.*;
Client myClient;

int i,j,R,m;
int M[] = {0,0,0,0,0,0,0,0,0};
int turn = 0;

int state = 4;

boolean R_p = false;
boolean entered = false;

void setup() {
  size(350, 350);
  noStroke();
  frameRate(30);
  background(0,0,0);

  myPort = new Serial(this, Serial.list()[0], 9600);
  
}

void draw(){
  fill(0,0,0);
  rect(250,330,350,350);
  textSize(20);
  fill(250,250,250);
  text(str(state),250,345);
  textSize(100);
  
  if (state == 0){
    wipe();
    turn = 0;
    state = 1;
  }
  else if (state == 1){
    i = 0;
    j = 0;
    R_p = false;
    if (myPort.available() == 0){
      entered = false;
    }
    while (myPort.available() > 0) {
      char in = myPort.readChar();
      entered = true;
      if (in == '1'){i = 1; j = 1;}
      else if (in == '2'){i = 2; j = 1;}
      else if (in == '3'){i = 3; j = 1;}
      else if (in == '4'){i = 1; j = 2;}
      else if (in == '5'){i = 2; j = 2;}
      else if (in == '6'){i = 3; j = 2;}
      else if (in == '7'){i = 1; j = 3;}
      else if (in == '8'){i = 2; j = 3;}
      else if (in == '9'){i = 3; j = 3;}
      else if (in == '*'){R_p = true;i=1;j=1;}
    }
    
    println(str(i)+","+str(j));

    //i = 1*int((25 < mouseX)&(mouseX<125)) + 2*int((125 < mouseX)&(mouseX<225)) + 3*int((225 < mouseX)&(mouseX<325));
    //j = 1*int((25 < mouseY)&(mouseY<125)) + 2*int((125 < mouseY)&(mouseY<225)) + 3*int((225 < mouseY)&(mouseY<325));
    i = i - 1;
    j = j - 1;
    m = (3*j + i)*int(i > -1)*int(j > -1) + (-1)*int((j < 0)|(i < 0));
    if (R_p){
      R = 1;
      m = 0;}
    else{
      R = int((0 < mouseX)&(mouseX<25)&(0 < mouseY)&(mouseY<25));}
    fill(0,0,0);
    rect(300,330,30,20);
    textSize(20);
    fill(250,250,250);
    if (R != 1){
      text(str(m),300,350);}
    else{
      text("R",300,350);}
    if (entered|((R==1)&mousePressed)){
      state = 2;
    }
  }
  else if (state == 2){
    if (R == 1){
      state = 0;}
    else if (turn == 0){
      if (m == -1){
        state = 1;
      }
      else if (M[m] != 0){
        state = 1;}
      else if ((mousePressed|entered)&(m != -1)){
        entered = false;
        if (M[m] == 0){
          draw_move(m,0);
          M[m] = 1;
          turn = 1;
        }
      }
    }
    else if (turn == 1){
      String out = "";
      for (int k = 0;k<8;k++){
        out = out + str(M[k])+",";}
      out = out + str(M[8]);
      myClient.write(out);
      
      while (myClient.available() < 2) {}
      String comp = myClient.readString();
      
      turn = 0;
      state = 1;
            
      if (comp.charAt(0) == 'G'){
        draw_move(int(str(comp.charAt(1))),1);
        M[int(str(comp.charAt(1)))] = -1;
      }
      else if (comp.charAt(0) == 'A'){
        draw_move(int(str(comp.charAt(1))),1);
        M[int(str(comp.charAt(1)))] = -1;
        fill(190,20,20);
        textSize(100);
        text("LOSE",100,150);
        state = 3;
      }
      else if (comp.charAt(0) == 'B'){
        fill(20,190,20);
        textSize(100);
        text("WIN",150,150);
        state = 3;
      }
      else if (comp.charAt(0) == 'C'){
        fill(20,20,190);
        textSize(100);
        text("DRAW",40,150);
        state = 3;
      }
    }  
  }
  else if (state == 7){
    if (!mousePressed){
      state = 0;} 
  }
  else if (state == 6){ 
    if (mousePressed){
      state = 7;} 
  }
  else if (state == 3){
    long timer = millis();
    while (millis()-timer < 2000){} 
    state = 0;
  }
  else if (state == 5){
    myClient = new Client(this, "127.0.0.1",15002);
    
    while (myClient.available() < 5) { }
    String correct = myClient.readString(); 
    myClient.write("check");

    state = 0;
  }
  else if (state == 4){
    textSize(32);
    fill(160,30,50);
    text("Loading Brain...",20,100);
    state = 5;
  }
}

void draw_move(int m,int p){
  int i = m%3;
  int j = (m-i)/3;

  textSize(100);
  if (p == 0){
    text("X",25+19+100*i,25+100-12+100*j);}
  if (p == 1){
    text("O",25+19+100*i,25+100-12+100*j);
  }
}

void wipe(){
  for (int k = 0; k < 9; k++){
    M[k] = 0;
  }
  clear();
  background(0,0,0);
  fill(255,0,0);
  rect(0,0,25,25);

  fill(180,10,180);
  rect(25,25,100,100);
  fill(180,110,10);
  rect(125,25,100,100);
  fill(180,190,10);
  rect(225,25,100,100);

  fill(180,10,110);
  rect(25,125,100,100);
  fill(100,50,180);
  rect(125,125,100,100);
  fill(90,50,50);
  rect(225,125,100,100);

  fill(80,200,110);
  rect(25,225,100,100);
  fill(100,250,150);
  rect(125,225,100,100);
  fill(190,150,150);
  rect(225,225,100,100);
  
  stroke(250,250,250);
  strokeWeight(3);
  line(125,25,125,325);
  line(225,25,225,325);
  line(25,125,325,125);
  line(25,225,325,225);
  line(25,25,25,325);
  line(25,325,325,325);
  line(325,325,325,25);
  line(325,25,25,25);
  noStroke();
  
}

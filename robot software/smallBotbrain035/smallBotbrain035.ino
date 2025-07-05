/*
Murin Bot Brain v0.2
Implements Murin Machine Learning algorithm in a compact
form on a microcontroller-driven robot.

Learning signal is internal- based on resolving sensor-derived
conditions, such as being blocked by an obstacle.

Uses Gyro for direction orienting,
IR sensor for obstacle avoidance,
Optical Tach for speed sensing,
Servo for steering,
Gearbox attachment for drive

  Implements simplified state-action learning,
  basic SA table and update,
  simplified action system,
  flag setting for states and actions,
  tight loop measurements
 
     
  SA key:
      A  FR  FC  FL  BR  BC  BL   S
      X   1  2   3   4   5   6    7
S   X
UHx 1     Q   Q   Q   Q   Q   Q   Q
UHx 2     Q   Q   Q   Q   Q   Q   Q
UHx 3     Q   Q   Q   Q   Q   Q   Q
UHx 4     Q   Q   Q   Q   Q   Q   Q
UHx 5     Q   Q   Q   Q   Q   Q   Q
UHx 6     Q   Q   Q   Q   Q   Q   Q
...
  x- RLC
  state-action state: (blocked/stuck/prior movement,drive/turn)
  note: avoiding  -  0/1       -> UO
        stuck     -  0/1       -> HD
        drivemode -  1/2/3      -> FBS
        drivepos  -  34/54/74  -> RLC
        States: 6x4 + 4 = 28
  */

//Required libraries
#include <Wire.h>
#include <LSM303.h>
#include <L3G.h>
#include <Servo.h>

//device objects
Servo driveservo;
L3G gyro;
LSM303 compass;

//state/action vars
int drivepos = 54;  //34-R,54-C,74-L
int drivemode = 3;  //1-fwd,2-bwd,3-stp
int avoiding = 0;   //0-unobstructed;1-obstructed
int stuck = 0;      //0-free,1-stuck
int count;

//filter vars
float hpfAlpha = .005;
float lpfAlpha = .005;
float gyroYim = 0;
float gyroXim = 0;
float theta = 0;
float velocimeter = 0;
float Vim = 0;
float disp = 0;

//timing vars
float milicheck = 0;
long dt = 0;
float DT = 0;
long timer = 0;
long t50ms = 0;
long t200ms = 0;
long t1s = 0;

//SA vars
int priorSAs[3];
int priorAct = 0;
int SA[28][7];
int alpha = 0.9;
int gamma = 0.1;

//Sensor update function
int sensorUpdate(){
  //gets gyro data
  dt = millis() - milicheck;
  DT = 1.0*dt;
  if (DT > 100){
    gyro.read();
    if ((gyro.g.z > -15.0)|(gyro.g.z< -150.0)){
      gyroYim = hpfAlpha*gyroYim + hpfAlpha*(gyro.g.z-gyroXim);
      theta = theta + (DT/500.0)*gyroYim;}
    milicheck = millis();}
  return 0;}

//Turn function  
void turn(){
  //Drive servo physically turns wheels- car style steering
  driveservo.write(drivepos);}
  
//Drive function
void drive(int cmd){
   drivemode = cmd; //Set mode state

   if (cmd == 1){ //Motors ahead
    digitalWrite(3,LOW);
    digitalWrite(2,HIGH);}
  if (cmd == 2){ //Motors reverse
    digitalWrite(2,LOW);
    digitalWrite(3,HIGH);}
  if (cmd == 3){ //Stop
    digitalWrite(2,LOW);
    digitalWrite(3,LOW);}}
  
void setup(){
  Serial.begin(9600); //Start serial port
  Wire.begin(); //Start I2C port

  //Attach drive servo and set to initial position
  driveservo.attach(6);
  driveservo.write(drivepos);

  //Initialize gyroscopr
  if (!gyro.init()){ //If gyro init failed
    tone(4,125,500); //Frequency signal to speaker
    delay(1000); //Wait a second
    digitalWrite(4,LOW); //Set control pin low
    while (1);//stop execution
    }
  gyro.enableDefault(); //Otherwise start up gyro
  }

//Main loop
void loop(){

  //timing loops
  if (millis()-t200ms > 200){ //200ms routines
    //UO status:
    int dist = analogRead(A0);

    //Detecting obstacles flag set
    if ((dist >= 150)){
      avoiding = 1;}
    if ((dist < 100) & (avoiding == 1)){ //Clearing avoidance flag
      avoiding = 0;}
    t200ms = millis();}//reset check loop

  if (millis()-t50ms > 50){ //50ms routines

    //HD status foot-work: 
    double V = analogRead(A2);
    V = int(.05*V + .95*velocimeter); //Read speed sensor
    if (((velocimeter > V)&(Vim == 0))|((velocimeter < V)&(Vim == 1))){
      Vim = 1-Vim;
      count++;}

    //distance (cm) ~= 0.418*count
    velocimeter = V;
    sensorUpdate(); //Update sensors
    //use theta/distance here to compute positioning data
    //Note: use only change in count!
    t50ms = millis(); //reset timer
    }

  if (millis()-t1s > 1000){ //1s loop
    //QSA

    //compute HD status- sensing @20/s; comp @1/s
    if ((drivemode != 3)&(count == 0)){
      stuck = 1;} //detect if stuck- not moving but drive on
    else{
      stuck = 0;} //Clear stuck flag otherwise
    count = 0; //reset velocimeter count
    
    //compute State from Sb, Ap
    int Sb = 14*(1-stuck);	//State and action are stride-encoded functions
    Sb = Sb + 7*(1-avoiding);	// of the general operation flags
    int Ap = 3*(drivemode - 1);
    if (Ap < 6){
      Ap = Ap + (drivepos - 34)/20;} 
    int S = Sb + Ap; //Augmented state: observed and action together
    
    //compute prior reward (use Sp->S)
    int Sp = priorSAs[0];
    int Sps = (Sp-Sp%7);
    //Sps: same format as Sb
    /* reward table: based on aim to reward correction of avoid and stuck states
      Sb 0  7  14  21
    Sp
     0  -2  1   1   2
     7  -2 -1   0   1
    14  -2  1  -1   1
    21   0  0   0   0
    */
    int RWD[4][4] = {{-2,1,1,2},{-2,1,0,1},{-2,1,-1,1},{0,0,0,0}};
    int reward = RWD[Sps/7][Sb/7]; 
    
    //for all 3 in priosSAs:     
    //update Q table- use max Q of state to which Sp,Ap lead:
    //i.e. the current state
    //possible to update Q after deciding on next action
    //use Q of that action instead of Qmax <- option
    //also, note decay term on earlier SAs
    
    //for Sn-1
    float Qmax = SA[S][0];
    for (int ind = 1; ind < 7; ind++){
      Qmax = max(Qmax,SA[S][ind]);}  
    float Q = SA[Sp][Ap] + alpha*(reward + gamma*Qmax);
    SA[Sp][Ap] = Q;
    
    //for Sn-2
    int SN2 = priorSAs[1];
    int AP2 = priorSAs[0]%7;
    float Q2 = SA[SN2][AP2] + (0.5)*alpha*(reward + gamma*Qmax);
    SA[SN2][AP2] = Q2;
    
    //for sn-3
    SN2 = priorSAs[2];
    AP2 = priorSAs[1]%7;
    Q2 = SA[SN2][AP2] + (0.25)*alpha*(reward + gamma*Qmax);
    SA[SN2][AP2] = Q2;
    
    //policy for action
    float Qsum = 0;
    for (int ind = 0; ind < 7; ind++){
      Qsum = Qsum + SA[S][ind];} //State slice cumulative sum

    int rand = random(0,1000); //probability accuract to 3 decimals
    int Av = 0; //add up action value index and prob. cumsum until passing random value
    int gval = 0;
    while ((Av < 7)&(gval < rand)){
      gval = gval + abs(1000*((1+SA[S][Av])/(Qsum+7)));
      Av = Av + 1;}

    Serial.println(Av); //Report selected value over serial

    //Action
    drivemode = (Av - Av%3)/3; //Turn action into drivemode and position states
    if (drivemode != 3){
      drivepos = 34 + 20*(Av%3);}
    turn(); //set steering to new orientation
    drive(drivemode); //set drive to new drive mode
    
    //update priorSAs
    priorSAs[2] = priorSAs[1];
    priorSAs[1] = priorSAs[0];
    priorSAs[0] = S;
  
    //Update 1 second timer
    t1s = millis();}
    
}

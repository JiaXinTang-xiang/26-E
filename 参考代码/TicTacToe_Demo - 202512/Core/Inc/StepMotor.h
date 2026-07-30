#ifndef __STEPMOTOR_H__
#define __STEPMOTOR_H__
#define PositiveDir 1  //正方向运动方向
#define NegativeDir 0

#define DrivePaulseTime 200

#define StepMotorEnable 0  //低使能
#define StepMotorDisable 1

#define StepMotor_None 0
#define StepMotorX 1 //X轴
#define StepMotorY 2 //Y轴
#define StepMotorZ 3 //Z轴

//#define MAXPosX 1150//行程范围内X轴步进电机运动的最大范围，从归零位置运动到最右边，需要的脉冲数
//#define MAXPosX 2200//行程范围内X轴步进电机运动的最大范围，从归零位置运动到最右边，需要的脉冲数
//#define MAXPosY 940//Y轴步进电机运动的最大范围，从归零位置运动到最右边，需要的脉冲数
#define MAXPosX 2350//行程范围内X轴步进电机运动的最大范围，从归零位置运动到最右边，需要的脉冲数202512修改
#define MAXPosY 1350//Y轴步进电机运动的最大范围，从归零位置运动到最右边，需要的脉冲数202512修改
#define MAXPosZ 400




#define CaliPosX1 0
#define CaliPosX2 MAXPosX
#define CaliPosY1 0
#define CaliPosY2 MAXPosY

#define CaliReverseShiftX 10//归零后再向正方向移动N个脉冲，用归零开关来定位
#define CaliReverseShiftY 8 //归零后再向正方向移动N个脉冲，用归零开关来定位，如果弹簧太松，可以适当增加数值

//井字棋9宫格坐标 [格子编号][X,Y]
static const unsigned short ChessGrid[9][2] =
{
    {392, 225},   // [0] 左上
    {1175, 225},  // [1] 中上
    {1958, 225},  // [2] 右上
    {392, 675},   // [3] 左中
    {1175, 675},  // [4] 正中
    {1957, 675},  // [5] 右中
    {392, 1125},  // [6] 左下
    {1175, 1125}, // [7] 中下
    {1958, 1125}, // [8] 右下
};//111111111111111111111111
//井字棋9宫格坐标 [格子编号][X,Y]
static const unsigned short Chessplace[10][2] =
{
    {230, 330},   // [0] 
    {230, 503},  // [1] 
    {230, 675},  // [2] 
    {230, 847},   // [3] 
    {230, 1020},  // [4] 
    {2030, 330},  // [5] 
    {2030, 503},  //[6] 
    {2030, 675}, // [7] 
    {2030, 847}, // [8] 
	{2030, 1020},
};//111111111111111111111111


static const unsigned short Chessboard[9][2] =
{
    {780,  433},  // 左上1号
    {1090, 433},  // 中上2号
    {1400, 433},  // 右上3号
    {780,  664},  // 左中4号
    {1090, 664},  // 正中5号
    {1400, 664},  // 右中6号
    {780,  895},  // 左下7号
    {1090, 895},  // 中下8号
    {1400, 895},  // 右下9号
};


#define SHORT_PRESS_PLUSE 10 //短按一次按键步进电机移动的脉冲数，20细分同步带，可以移动N * 0.2mm
//2GT同步带减速比2mm齿
//16细分同步带，假设电机驱动方式为1脉冲步进距离为（1.8°*2mm*16/360°）=0.16mm，同步带转1圈需要走12.5个脉冲，同步带移动2mm
//20细分同步带，假设电机驱动方式为1脉冲步进距离为（1.8°*2mm*20/360°）=0.20mm，同步带转1圈需要走10个脉冲，同步带移动2mm


typedef unsigned char uint8_t;
typedef unsigned short     int uint16_t;
void DriveStepMotor(uint8_t WhichMotor,uint8_t Dir, uint16_t PulseNum);//哪个电机需要使能，运动多少个脉冲
void EnableStepMotor(uint8_t WhichMotor);//使能步进电机
void DisableStepMotor(uint8_t WhichMotor);//停止步进电机
void ResetStepMotor(void);//上电时步进电机复位
void ControlEM(uint8_t FlagOn_Off);//控制电磁铁开关，0：关，1：开
void PutDownTheChess(uint16_t LocationX, uint16_t LocationY);//将棋子放到指定坐标处
void TakeAndPutDownTheChess(uint16_t LocationX0, uint16_t LocationY0, uint16_t LocationX1, uint16_t LocationY1);//在指定位置取棋子，并放到指定坐标处
void TestMotor1(void);////测试平台的机械臂运动最大范围
void TestMotor2(void);//取棋子测试
#endif

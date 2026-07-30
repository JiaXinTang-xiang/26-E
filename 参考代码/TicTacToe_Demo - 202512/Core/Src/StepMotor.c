#include "StepMotor.h"
#include "main.h"
#include "stdio.h"
#include "tim.h"
uint16_t PulseNumOfStepMotorY = 0;
uint16_t PulseNumOfStepMotorX = 0;
uint16_t PulseNumOfStepMotorZ = 0;

uint8_t FlagDirOfStepMotorY = PositiveDir;
uint8_t FlagDirOfStepMotorX = PositiveDir;
uint8_t FlagDirOfStepMotorZ = PositiveDir;

extern TIM_HandleTypeDef htim2;

uint16_t PosY = 0;
uint16_t PosX = 0;
uint16_t PosZ = 0;//步进电机当前位置
//2GT同步带减速比2mm齿
//16细分同步带，假设电机驱动方式为1脉冲步进距离为（1.8°*2mm*16/360°）=0.16mm，同步带转1圈需要走12.5个脉冲，同步带移动2mm
//20细分同步带，假设电机驱动方式为1脉冲步进距离为（1.8°*2mm*20/360°）=0.20mm，同步带转1圈需要走10个脉冲，同步带移动2mm
uint8_t FlagMotorYReset = 0;
uint8_t FlagMotorXReset = 0;
uint8_t FlagMotorZReset = 0;

uint8_t FlagReset = 0;//复位标志
uint8_t FlagMotorMove = 0;//电机运动标志，归零的时候不采集和计算。
void ResetStepMotor(void)
{
	FlagReset = 1;

	ControlEM(0);//关闭电磁铁
	HAL_Delay(DrivePaulseTime);
//	if(HAL_GPIO_ReadPin(GPIOB, PB14_LSY_Pin) == GPIO_PIN_RESET) FlagMotorYReset = 1;
	DriveStepMotor(StepMotorZ, NegativeDir, MAXPosZ);	//NegativeDir , PositiveDir
	while(FlagMotorZReset == 0)//抬升机械臂
	{
			if(PulseNumOfStepMotorZ == 0 )
			{
				FlagMotorZReset = 1;
				DisableStepMotor(StepMotorZ);
			}
			HAL_Delay(1);
	}

	if(HAL_GPIO_ReadPin(GPIOB, PB15_LSX_Pin) == GPIO_PIN_RESET) FlagMotorXReset = 1;
	if(HAL_GPIO_ReadPin(GPIOA, PA6_LSZ_Pin) == GPIO_PIN_RESET) FlagMotorZReset = 1;
	DriveStepMotor(StepMotorY, NegativeDir, MAXPosY);

	DriveStepMotor(StepMotorX, NegativeDir, MAXPosX);

//X,Y轴同时归零
	while(FlagMotorYReset == 0 || FlagMotorXReset == 0  )//归零限位被压下，退出循环，可根据各轴状态来单独过归零
	{
		if(HAL_GPIO_ReadPin(GPIOB, PB15_LSX_Pin) == GPIO_PIN_RESET)
		{
			FlagMotorXReset = 1;
			DisableStepMotor(StepMotorX);
		}

		if(HAL_GPIO_ReadPin(GPIOB, PB14_LSY_Pin) == GPIO_PIN_RESET)
		{
			FlagMotorYReset = 1;
			DisableStepMotor(StepMotorY);
//			HAL_TIM_PWM_Stop_IT(&htim3,TIM_CHANNEL_3);//关闭Y_STEP,机械臂水平移动的Y轴步进电机
		}
		HAL_Delay(1);

	}
	HAL_Delay(DrivePaulseTime);

//X和Y轴归零后，再移动离开归零点，避免使用限位开关松开时对后续操作的影响
	if(HAL_GPIO_ReadPin(GPIOB, PB15_LSX_Pin) == GPIO_PIN_SET) FlagMotorXReset = 0;
	if(HAL_GPIO_ReadPin(GPIOA, PA6_LSZ_Pin) == GPIO_PIN_SET) FlagMotorZReset = 0;
	DriveStepMotor(StepMotorY, PositiveDir, MAXPosY);
	DriveStepMotor(StepMotorX, PositiveDir, MAXPosX);
	while(FlagMotorYReset == 1 || FlagMotorXReset == 1  )//归零限位被压下，退出循环，可根据各轴状态来单独过归零
	{
		if(HAL_GPIO_ReadPin(GPIOB, PB15_LSX_Pin) == GPIO_PIN_SET)
		{
			FlagMotorXReset = 0;
			DisableStepMotor(StepMotorX);
		}

		if(HAL_GPIO_ReadPin(GPIOB, PB14_LSY_Pin) == GPIO_PIN_SET)
		{
			FlagMotorYReset = 0;
			DisableStepMotor(StepMotorY);
		}
		HAL_Delay(1);

	}

	HAL_Delay(DrivePaulseTime);
	Beep(DrivePaulseTime);
	HAL_Delay(DrivePaulseTime);
	FlagReset = 0;

	PulseNumOfStepMotorY = 0;
	PosY = 0;

	PulseNumOfStepMotorX = 0;
	PosX = 0;

	PulseNumOfStepMotorZ = 0;
	PosZ = 0;

	FlagMotorYReset = 0;
	FlagMotorXReset = 0;
	FlagMotorZReset = 0;

//	HAL_Delay(DrivePaulseTime);

}


void PutDownTheChess(uint16_t LocationX, uint16_t LocationY)//将棋子放到指定坐标处
{
	if(LocationX >= PosX)
		DriveStepMotor(StepMotorX, PositiveDir, LocationX - PosX);
	else
		DriveStepMotor(StepMotorX, NegativeDir, PosX - LocationX);

	if(LocationY >= PosY)
		DriveStepMotor(StepMotorY, PositiveDir, LocationY - PosY);
	else
		DriveStepMotor(StepMotorY, NegativeDir, PosY - LocationY);

	while( PulseNumOfStepMotorY > 0 || PulseNumOfStepMotorX > 0 ) HAL_Delay(1);//X和Y轴运动到位后,while循环需要加上延时否则会过快导致进入死循环
//		printf("AAA:%#x  BBB:%#x\r\n",PulseNumOfStepMotorY,PulseNumOfStepMotorX);
	HAL_Delay(DrivePaulseTime);

	DriveStepMotor(StepMotorZ, PositiveDir, MAXPosZ);	//Z轴下降
	while( PulseNumOfStepMotorZ > 0 ) HAL_Delay(1);
	//		printf("C:%#x\r\n",PulseNumOfStepMotorZ);

	HAL_Delay(20);
	ControlEM(0);//关闭电磁铁
	HAL_Delay(20);//

  FlagMotorZReset = 0;

	ResetStepMotor();	//复位
}

void TakeAndPutDownTheChess(uint16_t LocationX0, uint16_t LocationY0, uint16_t LocationX1, uint16_t LocationY1)//在指定位置取棋子，并放到指定坐标处
{
	if(LocationX0 >= PosX)
		DriveStepMotor(StepMotorX, PositiveDir, LocationX0 - PosX);
	else
		DriveStepMotor(StepMotorX, NegativeDir, PosX - LocationX0);

	if(LocationY0 >= PosY)
		DriveStepMotor(StepMotorY, PositiveDir, LocationY0 - PosY);
	else
		DriveStepMotor(StepMotorY, NegativeDir, PosY - LocationY0);

	while( PulseNumOfStepMotorY > 0 || PulseNumOfStepMotorX > 0 ) HAL_Delay(1);//X和Y轴运动到位后,while循环需要加上延时否则会过快导致进入死循环

	HAL_Delay(DrivePaulseTime);		//停顿一下

	DriveStepMotor(StepMotorZ, PositiveDir, MAXPosZ);	//Z轴下降
	while( PulseNumOfStepMotorZ > 0 ) HAL_Delay(1);

	HAL_Delay(20);
	ControlEM(1);//打开电磁铁,吸住棋子
	HAL_Delay(20);

	DriveStepMotor(StepMotorZ, NegativeDir, MAXPosZ);	//Z轴抬升
	while( PulseNumOfStepMotorZ > 0 ) HAL_Delay(1);

	HAL_Delay(DrivePaulseTime);		//停顿一下

	PutDownTheChess(LocationX1, LocationY1);//放到指定位置

}

void EnableStepMotor(uint8_t WhichMotor)//使能步进电机
{
	switch (WhichMotor)
  {
  	case StepMotorX:
			HAL_GPIO_WritePin(GPIOB, STEPX_EN_Pin, GPIO_PIN_RESET);
  		break;
  	case StepMotorY:
			HAL_GPIO_WritePin(GPIOB, STEPY_EN_Pin, GPIO_PIN_RESET);
  		break;
  	case StepMotorZ://对于Z轴步进电机，可以只通过控制驱动器的使能端来控制
			HAL_GPIO_WritePin(GPIOB, STEPZ_EN_Pin, GPIO_PIN_RESET);
  		break;
  	default:
  		break;
  }
	FlagMotorMove = 1;

}
void DisableStepMotor(uint8_t WhichMotor)//停止步进电机
{
	switch (WhichMotor)
  {
  	case StepMotorX:
			HAL_GPIO_WritePin(GPIOB, STEPX_EN_Pin, GPIO_PIN_SET);
  		break;
  	case StepMotorY:
			HAL_GPIO_WritePin(GPIOB, STEPY_EN_Pin, GPIO_PIN_SET);
  		break;
  	case StepMotorZ://Z轴电机
			HAL_GPIO_WritePin(GPIOB, STEPZ_EN_Pin, GPIO_PIN_SET);
  		break;
  	default:
  		break;
  }
	FlagMotorMove = 0;
}

void DriveStepMotor(uint8_t WhichMotor, uint8_t Dir, uint16_t PulseNum)
{//哪个电机需要使能，运动多少个脉冲
	switch (WhichMotor)
  {
  	case StepMotorX:
		{
			EnableStepMotor(StepMotorX);//使能X轴
			if(Dir == PositiveDir)//正方向
				HAL_GPIO_WritePin(GPIOA, STEPX_DIR_Pin, GPIO_PIN_RESET);
			else
				HAL_GPIO_WritePin(GPIOA, STEPX_DIR_Pin, GPIO_PIN_SET);

			PulseNumOfStepMotorX = PulseNum;
			FlagDirOfStepMotorX = Dir;
  		break;
		}
  	case StepMotorY:
		{
			EnableStepMotor(StepMotorY);//使能Y轴
			if(Dir == PositiveDir)//正方向
				HAL_GPIO_WritePin(GPIOB, STEPY_DIR_Pin, GPIO_PIN_RESET);
			else
				HAL_GPIO_WritePin(GPIOB, STEPY_DIR_Pin, GPIO_PIN_SET);


			PulseNumOfStepMotorY = PulseNum;
			FlagDirOfStepMotorY = Dir;
  		break;
		}

  	case StepMotorZ:
		{
			EnableStepMotor(StepMotorZ);//使能Z轴
			if(Dir == PositiveDir)//正方向
				HAL_GPIO_WritePin(GPIOA, STEPZ_DIR_Pin, GPIO_PIN_RESET);
			else
				HAL_GPIO_WritePin(GPIOA, STEPZ_DIR_Pin, GPIO_PIN_SET);

			PulseNumOfStepMotorZ = PulseNum;

			FlagDirOfStepMotorZ = Dir;
  		break;
		}
  	default:
  		break;

  }

}

void ControlEM(uint8_t FlagOn_Off)//控制电磁铁开关，0：关，1：开
{
	if(FlagOn_Off == 1)
	{
		HAL_GPIO_WritePin(GPIOB, EM_Pin, GPIO_PIN_SET);//打开电磁铁
		HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);//亮LED
	}
	else
	{
		HAL_GPIO_WritePin(GPIOB, EM_Pin, GPIO_PIN_RESET);//关闭电磁铁
		HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_SET);//灭LED
	}
}


void TestMotor1(void)//测试平台的机械臂运动最大范围
{
	DriveStepMotor(StepMotorX, PositiveDir, MAXPosX);
	while( PulseNumOfStepMotorX > 0 ) HAL_Delay(1);//X轴运动到位后
	HAL_Delay(DrivePaulseTime);		//停顿一下

	DriveStepMotor(StepMotorY, PositiveDir, MAXPosY);
	while( PulseNumOfStepMotorY > 0 ) HAL_Delay(1);//Y轴运动到位后
	HAL_Delay(DrivePaulseTime);		//停顿一下

	DriveStepMotor(StepMotorX, NegativeDir, MAXPosX);
	while( PulseNumOfStepMotorX > 0 ) HAL_Delay(1);//X轴运动到位后
	HAL_Delay(DrivePaulseTime);		//停顿一下

	DriveStepMotor(StepMotorY, NegativeDir, MAXPosY);
	while( PulseNumOfStepMotorY > 0 ) HAL_Delay(1);//Y轴运动到位后
	HAL_Delay(DrivePaulseTime);		//停顿一下

	ResetStepMotor();//各轴步进电机复位

}

void TestMotor2(void)//取棋子测试
{

	TakeAndPutDownTheChess(CaliPosX1, CaliPosY1, CaliPosX2, CaliPosY2);//在指定位置取棋子，并放到指定坐标处
	HAL_Delay(200);
	TakeAndPutDownTheChess(CaliPosX2, CaliPosY2, CaliPosX2, CaliPosY1);//在指定位置取棋子，并放到指定坐标处
	HAL_Delay(200);
	TakeAndPutDownTheChess(CaliPosX2, CaliPosY1, CaliPosX1, CaliPosY2);//在指定位置取棋子，并放到指定坐标处
	HAL_Delay(200);
	TakeAndPutDownTheChess(CaliPosX1, CaliPosY2, CaliPosX1, CaliPosY1);//在指定位置取棋子，并放到指定坐标处
	HAL_Delay(200);
}

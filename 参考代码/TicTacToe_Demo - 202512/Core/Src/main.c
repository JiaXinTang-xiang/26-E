/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
	*桂林电子科技大学信息与通信学院 梁勇
  ******************************************************************************
本程序为STM32F103C8T6的板HAL库版本
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "key.h"
#include "stdio.h"
#include "oled.h"
//#include "bmp.h"
#include "StepMotor.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
//***********以下变量在key.c中定义
extern KEY_NAME KeyNameNow;//当前按下的是哪个按键;
extern KEY_NAME KeyNameBefore;//记录上一次按下的是哪个按键
extern KEY_STATE KeyState;//按键状态，长按或短按的返回值为按下时的状态
extern KEY_STATE KeyStateBefore;//按键之前的历史状态，长按或短按的返回值为按下时的状态
extern MENU_NAME MenuName;//当前的菜单选项
extern PRESS_KEY PressTheKeyAction;

extern uint16_t PulseNumOfStepMotorX;
extern uint16_t PulseNumOfStepMotorY;//步进电机运动的脉冲数
extern uint16_t PulseNumOfStepMotorZ;
extern uint8_t FlagDirOfStepMotorX;//步进电机方向
extern uint8_t FlagDirOfStepMotorY;
extern uint8_t FlagDirOfStepMotorZ;
extern uint16_t PosX;//步进电机位置
extern uint16_t PosY;
extern uint16_t PosZ;//步进电机当前位置
extern uint8_t FlagMotorZReset;//Z轴复位标志
extern uint8_t FlagLEDn;//为41时表示LED正在被操作，为40时表示LED空闲操作
extern uint8_t FlagReset ;//复位标志
extern uint8_t FlagMotorHomeing;
extern uint8_t UART1_Rx_flg;//串口接收数据的中断标志
extern uint16_t PosBuf0[3];//取棋子的电源位置坐标
extern uint16_t PosBuf1[3] ;//放棋子的目标位置坐标
extern uint8_t RecBuf[];//串口接收缓冲
uint16_t BeepCurrentMillis = 0;//蜂鸣器发声计时
uint16_t BeepTime = 0;//蜂鸣器响的时长
uint8_t FlagBeep = 0;//蜂鸣器工作标志

uint8_t FlagEM = 0;//电磁铁开关标志
//extern uint8_t FlagEnabeMotorC_Tim2;


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

//	uint8_t MenuName = MENU_MAIN_WalkAside;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_ADC1_Init();
  MX_TIM4_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */

	HAL_ADCEx_Calibration_Start(&hadc1);//开始ADC之前校准
	__HAL_TIM_CLEAR_IT(&htim4, TIM_IT_UPDATE);//清除定时器的更新中断标志
	HAL_TIM_Base_Start_IT(&htim4);//使能步进电机定时器4
	Beep(100);
	HAL_Delay(100);
	OLED_Init();
	OLED_ColorTurn(0);//0正常显示，1 反色显示
  OLED_DisplayTurn(0);//0正常显示 1 屏幕翻转显示
	//OLED_ShowMenu(MenuName);

	HAL_TIM_PWM_Start_IT(&htim1,TIM_CHANNEL_1);//启动X_STEP,底座支撑臂水平移动的X轴步进电机
	HAL_TIM_PWM_Start_IT(&htim3,TIM_CHANNEL_3);//启动Y_STEP,机械臂水平移动的Y轴步进电机
	HAL_TIM_PWM_Start_IT(&htim2,TIM_CHANNEL_2);//启动Z_STEP,机械臂上下移动的Z轴步进电机

	ResetStepMotor();//各轴步进电机复位

	HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);//点亮LED灯1亮
	HAL_GPIO_WritePin(LED2_GPIO_Port, LED2_Pin, GPIO_PIN_RESET);
	HAL_Delay(400);
	HAL_GPIO_TogglePin(LED1_GPIO_Port, LED1_Pin);
	HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
//

//	HAL_UART_Transmit(&huart1,RecBuf,17,10); //显示运行中数值 ,调试时可以打开使用

  while (1)
  {

		if(UART1_Rx_flg ==1)//串口发送控制命令，单片机串口接收到17个字节数据后
		{
			if(USART1_RecCommand() ==1)//命令解析取数据
			{
				TakeAndPutDownTheChess(PosBuf0[0], PosBuf0[1], PosBuf1[0], PosBuf1[1]);//在指定位置取棋子，并放到指定坐标处
			}
			UART1_Rx_flg = 0;
			HAL_UART_Receive_IT(&huart1, RecBuf, 17);//等待下一次的接收
		}


		GetKeyStatue();//获取按键状态值，放在大循环中处理占用时间短的话，按键识别不到按键抬键

//        // TakeAndPutDownTheChess(Chessboard[0][0], Chessboard[0][1], Chessboard[1][0], Chessboard[1][1]);
//		// TakeAndPutDownTheChess(Chessboard[2][0], Chessboard[2][1], Chessboard[3][0], Chessboard[3][1]);
//		// TakeAndPutDownTheChess(Chessboard[4][0], Chessboard[4][1], Chessboard[5][0], Chessboard[5][1]);
//		// TakeAndPutDownTheChess(Chessboard[6][0], Chessboard[6][1], Chessboard[7][0], Chessboard[7][1]);
//		// TakeAndPutDownTheChess(Chessboard[8][0], Chessboard[8][1], Chessboard[0][0], Chessboard[0][1]);















// 		if(PressTheKeyAction != PressNoneKey) //有按键按下时按键状态的改变
// 		{
// //			printf("PressTheKeyAction:%#x\r\n",PressTheKeyAction);

// 			switch (MenuName)//根据当前按下的菜单项，和读取的按键状态进行判断处理
// 			{
// 				case MENU_MAIN_WalkAside://主菜单的巡边 选项
// 					switch (PressTheKeyAction)//按键处理
// 					{
// 						case PressKey_Down://短按下键
// 						case PressKey_Right://短按右键
// 							MenuName = MENU_MAIN_MoveTheChess ;
// 							break;
// 						case PressKey_Up://短按上键
// 						case PressKey_Left://短按左键
// 							MenuName = MENU_MAIN_Z_Axis0 ;
// 							break;

// 						case PressKey_Mid://短按中键
// 							OLED_ShowChinese(0,16,15,16,1);//显示实心圆，表示正在写入人工操作
// 							OLED_Refresh();
// 							TestMotor1();////测试平台的机械臂运动最大范围
// 							OLED_ShowChinese(0,16,49,16,1);//
// 							OLED_Refresh();
// 							break;

// 						default:
// 							break;
// 					}
// 					break;
// 				case MENU_MAIN_MoveTheChess://主菜单的搬棋 选项
// 					switch (PressTheKeyAction)//按此菜单的2选项
// 					{
// 						case PressKey_Down://短按下键
// 						case  PressKey_Right://短按右键
// 							MenuName ++;
// 							break;
// 						case PressKey_Left://短按左键
// 						case PressKey_Up://短按上键
// 							MenuName --;
// 							break;

// 						case PressKey_Mid://短按中键
// 							OLED_ShowChinese(0,32,15,16,1);//显示实心圆，表示正在写入人工操作
// 							OLED_Refresh();
// 							TestMotor2();////测试平台的机械臂夹取棋子
// 							OLED_ShowChinese(0,32,49,16,1);//
// 							OLED_Refresh();

// 							break;
// 						default:
// 							break;
// 					}
// 					break;

// 				case MENU_MAIN_Z_Axis0://主菜单的Z轴 电动工具
// 					switch (PressTheKeyAction)//
// 					{
// 						case PressKey_Down://短按下键
// 						case  PressKey_Right://短按右键
// 							MenuName = MENU_MAIN_WalkAside;
// 							break;
// 						case PressKey_Up://短按上键
// 						case PressKey_Left://短按左键
// 							MenuName = MENU_MAIN_MoveTheChess;
// 							break;
// 						case PressKey_Mid://短按中键
// 							MenuName = MENU_ZAxis;
// 							break;
// 						default:
// 							break;

// 					}
// 					break;

// 				//*******************以下是Z轴的子菜单处理*******************************
// 				case MENU_ZAxis://以下是Z轴的子菜单处理
// 				case MENU_ZAxis_SHORT_PRESS_UP://在Z轴向上移动
// 				case MENU_ZAxis_SHORT_PRESS_DOWN://在Z轴向下移动
// 				case MENU_ZAxis_LONG_PRESS_UP://在Z轴向上移动
// 				case MENU_ZAxis_LONG_PRESS_DOWN://在Z轴向下移动
// 				case MENU_ZAxis_SHORT_PRESS_EM_ON://在Z轴向上移动
// 				case MENU_ZAxis_SHORT_PRESS_EM_OFF://在Z轴向下移动
// 					switch (PressTheKeyAction)
// 					{
// 						case PressKey_Up://短按上键
// 							if(PosZ > 0)
// 							{
// 								DriveStepMotor(StepMotorZ,NegativeDir,SHORT_PRESS_PLUSE);
// 								MenuName = MENU_ZAxis_SHORT_PRESS_UP;
// 							}
// 							break;
// 						case PressKey_Down://短按下键
// 							if(PosZ < MAXPosZ)
// 							{

// 								DriveStepMotor(StepMotorZ,PositiveDir,SHORT_PRESS_PLUSE);
// 								MenuName = MENU_ZAxis_SHORT_PRESS_DOWN;
// 							}
// 							break;
// 						case  PressAndHoldKey_Up://长按上键
// 							if(PosZ > 0)
// 							{
// 								DriveStepMotor(StepMotorZ,NegativeDir,SHORT_PRESS_PLUSE * 4);
// 								MenuName = MENU_ZAxis_LONG_PRESS_UP;
// 							}
// 							break;
// 						case  PressAndHoldKey_Down://长按下键
// 							if(PosZ < MAXPosZ)
// 							{
// 								DriveStepMotor(StepMotorZ,PositiveDir,SHORT_PRESS_PLUSE * 4);
// 								MenuName = MENU_ZAxis_LONG_PRESS_DOWN;
// 							}
// 							break;
// 						case PressKey_Mid://短按下键
// 							if(FlagEM == 0)
// 							{
// 								FlagEM = 1;
// 								MenuName = MENU_ZAxis_SHORT_PRESS_EM_ON;
// 								ControlEM(1);//打开电磁铁

// 							}
// 							else
// 							{
// 								FlagEM = 0;
// 								MenuName = MENU_ZAxis_SHORT_PRESS_EM_OFF;
// 								ControlEM(0);//关闭电磁铁
// 							}
// 							break;
// 						case PressKey_Left://短按左键
// 						case PressKey_Right://短按右键
// 						case PressKey_Side://短按侧键
// 						case PressAndHoldKey_Side://长按侧键
// 							MenuName = MENU_MAIN_Z_Axis0;//退出Z轴控制
// 							ControlEM(0);//关闭电磁铁
// 							DriveStepMotor(StepMotorZ,NegativeDir,MAXPosZ);
// 							break;

// 						default:
// 							break;
// 					}
// 					break;

// 				default:
// 					break;

// 				}
// 				OLED_ShowMenu(MenuName);
// 			}

		}//////////////////////////////////////////////////////////////////////////////////////////


    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */




  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void HAL_TIM_PWM_PulseFinishedCallback(TIM_HandleTypeDef *htim)
{

		if( htim ->Instance == TIM1  )//TIM1_CH2为X轴
		{

			if(FlagReset == 0)//在归零时，此中断还在运行，归零以限位开关为准，因此归零时不能用下面的代码
			{
				if(PulseNumOfStepMotorX >0)//TIM1_CH2为X轴，当前要运动的步进电机脉冲数
				{
					PulseNumOfStepMotorX--;//运动脉冲数减一
					if(FlagDirOfStepMotorX == PositiveDir && PosX <= MAXPosX)
						PosX++;
					else if(FlagDirOfStepMotorX == NegativeDir )
						PosX--;
				}
				else
				{
					DisableStepMotor(StepMotorX);

	//				printf("NumY:%#x\r\n",PulseNumOfStepMotorX);
	//				printf("PosX:%#x\r\n",PosX);
				}
	//
			}
		}
		if( htim ->Instance == TIM3  )//TIM3_CH3为Y轴
		{
			if(FlagReset == 0)//在归零时，此中断还在运行，归零以限位开关为准，因此归零时不能用下面的代码
			{

					if(PulseNumOfStepMotorY >0)//当前要运动的步进电机脉冲数
					{
						PulseNumOfStepMotorY--;//运动脉冲数减一
						if(FlagDirOfStepMotorY == PositiveDir && PosY <= MAXPosY)
							PosY++;
						else if(FlagDirOfStepMotorY == NegativeDir  )
							PosY--;
					}
					else
					{
						DisableStepMotor(StepMotorY);
		//				printf("NumX:%#x\r\n",PulseNumOfStepMotorY);
		//				printf("PosY:%#x\r\n",PosY);
					}
			}
		}
		if( htim ->Instance == TIM2 )//TIM2_CH3为Z轴
		{
			if(PulseNumOfStepMotorZ >0)//当前要运动的步进电机脉冲数
			{
				PulseNumOfStepMotorZ--;//运动脉冲数减一
				if(FlagDirOfStepMotorZ == PositiveDir && PosZ <= MAXPosZ)
					PosZ++;
				else if(FlagDirOfStepMotorZ == NegativeDir && PosZ > SHORT_PRESS_PLUSE)
				{
					PosZ--;
				}

			}
			else
			{
				DisableStepMotor(StepMotorZ);
			}


		}
}

int fputc(int ch,FILE*f)
{
	HAL_UART_Transmit(&huart1,(uint8_t*)&ch,1,HAL_MAX_DELAY);
	return ch;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)//ADC回调函数，根据中断对应的ADC值
{

	uint16_t ADC0_Value = 0;//ADC的读值

	KeyNameBefore = KeyNameNow;//把前1次读到的按键值
	//**********判断按键的值*********************************************
	ADC0_Value = HAL_ADC_GetValue(&hadc1);//获取ADc转换的数据，其值为12位
	//根据按键按下时的ADC值取中间值，作为阈值，判断当前按下的是哪个按键
	if (ADC0_Value<561)////侧按键的按键值
		KeyNameNow = SIDE_KEY;
	else if(ADC0_Value < 2388)
		KeyNameNow = UP_KEY;
	else if(ADC0_Value < 2900)
		KeyNameNow = LEFT_KEY;
	else if(ADC0_Value < 3173)
		KeyNameNow =  MID_KEY;
	else if(ADC0_Value < 3345)
		KeyNameNow = RIGHT_KEY;
	else if(ADC0_Value < 3753)
		KeyNameNow = DOWN_KEY;
	else
		KeyNameNow = NONE_KEY;

	JudgeKeyByStateMachine();//判断按键处理
}

void HAL_TIM_PeriodElapsedCallback( TIM_HandleTypeDef *htim)
{

	if(FlagBeep == 1)
	{
		BeepCurrentMillis++;
		if(BeepCurrentMillis >= BeepTime)
		{
			FlagBeep = 0;
			HAL_GPIO_WritePin(Beep_GPIO_Port, Beep_Pin, GPIO_PIN_RESET);//关闭蜂鸣器
		}
	}
	if( htim ->Instance == TIM4 )
	{
		HAL_ADC_Start_IT(&hadc1);//启动ADC转换

	}
}

void Beep(uint16_t BeepDurtion)//蜂鸣器发声
{
	FlagBeep = 1;//此处打开，由TIM4的1ms定时中断来关闭
	BeepTime = BeepDurtion;
	BeepCurrentMillis = 0;
	HAL_GPIO_WritePin(Beep_GPIO_Port, Beep_Pin, GPIO_PIN_SET);//打开蜂鸣器


}

// 中断回调函数
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART1)
	{
//		RecBuf[UART1_Rx_cnt++] = UART1_temp[0];
//		if (UART1_temp[0] == 0x0A)
//		{
//
//		}

		UART1_Rx_flg = 1;
	}
}


/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

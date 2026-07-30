#include "key.h"
#include "usart.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LONGPRESSED_TIME 1000	//长按判断，1000ms确认
#define SHORTPRESSED_TIME 10  //短按判断，10ms确认
#define SHORTPRESSED2TIMES_TIME 500  //双击的间隔，小于300ms确认为一次双击

KEY_NAME KeyNameNow = NONE_KEY;//当前按下的是哪个按键
KEY_NAME KeyNameBefore = NONE_KEY;//记录上一次按下的是哪个按键
KEY_NAME KeyName2TimesPressed_FirstTime = NONE_KEY;//双击的判断中，记录前一次按下的键值

KEY_STATE KeyState = KEY_CHECK;//按键状态，长按或短按的返回值为按下时的状态
KEY_STATE KeyStateBefore = KEY_CHECK;//按键之前的历史状态，长按或短按的返回值为按下时的状态



struct KEYSTATE_NAME WhichKey_State = {KEY_CHECK,NONE_KEY};
uint8_t FlagKeyPresed = 0;
PRESS_KEY PressTheKeyAction = PressNoneKey;

char*  FunctionKeyValueState2String(struct KEYSTATE_NAME KeyNameState)
{
	static char StrKeyNameState[10];
	switch(KeyNameState.KeyState)
	{
		case KEY_SHORT_PRESS_CONFIRMED:
			strcpy(StrKeyNameState,"S_P");
			break;
		case KEY_SHORT_RELEASE:
			strcpy(StrKeyNameState,"S_Re");
			break;
		case KEY_LONG_PRESS_CONFIRMED:
			strcpy(StrKeyNameState,"L_P");
			break;
		case KEY_LONG_RELEASE:
			strcpy(StrKeyNameState,"L_Re");
			break;
		default:
      break;
	}

	switch(KeyNameState.KeyName)
	{
    case DOWN_KEY:
			strcat(StrKeyNameState," D\r\n");
    	break;
    case RIGHT_KEY:
			strcat(StrKeyNameState," R\r\n");
    	break;
    case MID_KEY:
			strcat(StrKeyNameState," M\r\n");
    	break;
    case LEFT_KEY:
			strcat(StrKeyNameState," L\r\n");
    	break;
    case UP_KEY:
			strcat(StrKeyNameState," U\r\n");
    	break;
    case SIDE_KEY:
			strcat(StrKeyNameState," K\r\n");
    	break;
    default:
    	break;
	}
  return StrKeyNameState;
}

void GetKeyStatue(void)//PressTheKeyAction的值，放在大循环中处理判断用
{

	if(	(WhichKey_State.KeyState == KEY_SHORT_PRESS_CONFIRMED) && (FlagKeyPresed == 1))//确认某一个键短按，短按确认标志来防止重复识别
	{
		FlagKeyPresed = 0;//短按后被识别一次后要清除此标志，否则会重复识别

		switch (WhichKey_State.KeyName)
    {
    	case DOWN_KEY:
				PressTheKeyAction = PressKey_Down;//短按按下，按下键
//				printf("S_P D\r\n");
    		break;
    	case RIGHT_KEY:
				PressTheKeyAction = PressKey_Right;//短按按下，按右键
//				printf("S_P R\r\n");
    		break;
    	case MID_KEY:
				PressTheKeyAction = PressKey_Mid;//短按按下，按中键
//				printf("S_P M\r\n");
    		break;
    	case LEFT_KEY:
				PressTheKeyAction = PressKey_Left;//短按按下，按左键
//				printf("S_P L\r\n");
    		break;
    	case UP_KEY:
				PressTheKeyAction = PressKey_Up;//短按按下，按上键
//				printf("S_P U\r\n");
    		break;
    	case SIDE_KEY:
				PressTheKeyAction = PressKey_Side;//短按按下，按侧键
//			printf("S_P K\r\n");
    		break;
    	default:
				PressTheKeyAction = PressNoneKey;//没有按键
    		break;
    }
	}
	else if(WhichKey_State.KeyState == KEY_SHORT_RELEASE)//某一个键短按抬键,此时WhichKey_State.KeyState记录的是抬键前的按键值来判断使用后要清除
	{																										//此处这样设计是因为记录抬键前的键值和状态，需要2次才能被识别为稳定抬键，避免误触会被识别2次。
		switch (WhichKey_State.KeyName)										//第1次识别为影响到的抬键的按键状态，第2次才被识别为稳定抬键
    {
    	case DOWN_KEY:
				PressTheKeyAction = ReleaseKey_Down;//短按按下后抬键，按下键
//				printf("S_Re D\r\n");
    		break;
    	case RIGHT_KEY:
				PressTheKeyAction = ReleaseKey_Right;//短按按下后抬键，按右键
//				printf("S_Re R\r\n");
    		break;
    	case MID_KEY:
				PressTheKeyAction = ReleaseKey_Mid;//短按按下后抬键按中键
//				printf("S_Re M\r\n");
    		break;
    	case LEFT_KEY:
				PressTheKeyAction = ReleaseKey_Left;//短按按下后抬键，按左键
//				printf("S_Re L\r\n");
    		break;
    	case UP_KEY:
				PressTheKeyAction = ReleaseKey_Up;//短按按下后抬键，按上键
//				printf("S_Re U\r\n");
    		break;
    	case SIDE_KEY:
				PressTheKeyAction = ReleaseKey_Side;//短按按下后抬键：按侧键
//				printf("S_Re K\r\n");
    		break;
    	default:
    		break;
    }
		WhichKey_State.KeyState = KEY_CHECK;
		WhichKey_State.KeyName = NONE_KEY;//此处这样设计是因为记录抬键前的键值和状态，需要2次才能被识别为抬键，避免误触会被识别2次
	}
	else if(WhichKey_State.KeyState == KEY_LONG_PRESS_CONFIRMED)//某一个键处于长按状态，按键按下时的改变某状态的值
	{
		switch (WhichKey_State.KeyName)
    {
    	case DOWN_KEY:
				PressTheKeyAction = PressAndHoldKey_Down;//长按按下，按下键
//				printf("L_P D\r\n");
    		break;
    	case RIGHT_KEY:
				PressTheKeyAction = PressAndHoldKey_Right;//长按按下，按右键
//				printf("L_P R\r\n");
    		break;
    	case MID_KEY:
				PressTheKeyAction = PressAndHoldKey_Mid;//长按按下，按中键
//				printf("L_P M\r\n");
    		break;
    	case LEFT_KEY:
				PressTheKeyAction = PressAndHoldKey_Left;//长按按下，按左键
//				printf("L_P L\r\n");
    		break;
    	case UP_KEY:
				PressTheKeyAction = PressAndHoldKey_Up;//长按按下，按上键
//				printf("L_P U\r\n");
    		break;
    	case SIDE_KEY:
				PressTheKeyAction = PressAndHoldKey_Side;//长按按下，按侧键
//				printf("L_P K\r\n");
    		break;
    	default:
    		break;
    }
	}
	else if(WhichKey_State.KeyState == KEY_LONG_RELEASE)		//长按按键抬键时的状态，用此函数。WhichKey_State.KeyState记录的是前一次的状态来判断使用后要清除
	{																									//此处这样设计是因为记录抬键前的键值和状态，需要2次才能被识别为稳定抬键，避免误触会被识别2次。
		switch (WhichKey_State.KeyName)										//第1次识别为影响到的抬键的按键状态，第2次才被识别为稳定抬键
    {
     	case DOWN_KEY:
				PressTheKeyAction = ReleaseHoldedKey_Down;//长按按下后抬键，按下键
//				printf("L_Re D\r\n");
    		break;
    	case RIGHT_KEY:
				PressTheKeyAction = ReleaseHoldedKey_Right;//长按按下后抬键，按右键
//				printf("L_Re R\r\n");
    		break;
    	case MID_KEY:
				PressTheKeyAction = ReleaseHoldedKey_Mid;//长按按下后抬键按中键
//				printf("L_Re M\r\n");
    		break;
    	case LEFT_KEY:
				PressTheKeyAction = ReleaseHoldedKey_Left;//长按按下后抬键，按左键
//				printf("L_Re L\r\n");
    		break;
    	case UP_KEY:
				PressTheKeyAction = ReleaseHoldedKey_Up;//长按按下后抬键，按上键
//				printf("L_Re U\r\n");
    		break;
    	case SIDE_KEY:
				PressTheKeyAction = ReleaseHoldedKey_Side;//长按按下后抬键：按侧键
//				printf("L_Re K\r\n");
    		break;
    	default:
    		break;
    }
		WhichKey_State.KeyState = KEY_CHECK;
		WhichKey_State.KeyName = NONE_KEY;//此处这样设计是因为记录抬键前的键值和状态，需要2次才能被识别为抬键，避免误触会被识别2次

	}
	else
		PressTheKeyAction = PressNoneKey;
}

void JudgeKeyByStateMachine(void) //状态机判断按键状态，无按键、短按、短按抬键、长按、长按抬键
{

	static uint16_t KeyPressCounterOf1ms = 0;//1ms计数器

	//**********有按键按下后，定时累计*********************************************
	//ADC在TIM4中，每1ms执行一次，所以ADC中断中的这个函数，只要有按键按下时1ms加1
	//有键按下，或抬键时，以下代码将前按键加1ms，分成4个状态（情况）：
	//情况1：当前无按键，当前无按键
	//情况2：当前无按键，当前有按键 ,或有按键时有按键，注意：因为1ms很短，可能不够ADC采样时间，所以按键只和当前的上一次按键相同
	//情况3：当前有按键，当前无按键
	if( KeyNameNow == KeyNameBefore) //键值和前一次相同
	{
		if(KeyNameNow != NONE_KEY)
		{//#################情况2：当前无按键，当前有按键,或有按键时有按键##########################
			KeyPressCounterOf1ms++;

			//**********状态机判断是短按还是长按，并输出对应的信号处理*********************************************
			KeyStateBefore = KeyState;//记录之前的状态
			switch(KeyState)
			{
				case KEY_CHECK:
				{
					KeyState = KEY_SHORT_PRESS;//检测到有按键按下，就进入短按确认状态
					break;
				}
				case KEY_SHORT_PRESS://短按确认状态
				{
					if(KeyPressCounterOf1ms>=SHORTPRESSED_TIME)//按键稳定时间超过滤波时间SHORTPRESSED_TIME，确认短按成立，下次转入长按判断状态
					{
						KeyState = KEY_SHORT_PRESS_CONFIRMED;
						FlagKeyPresed = 1;//本次按键按下标志，主要用于防止在长按前会重复识别
						WhichKey_State.KeyName = KeyNameNow;
					}
					else//如果只是干扰信号，回到初始状态，重新检测状态
					{
						KeyState = KEY_CHECK;
						WhichKey_State.KeyName = NONE_KEY;
					}
					break;
				}
				case KEY_SHORT_PRESS_CONFIRMED://长按判断状态
				{
					if(KeyPressCounterOf1ms>=LONGPRESSED_TIME)
					{

						KeyPressCounterOf1ms = LONGPRESSED_TIME;	//按键时间超过LONGPRESSED_TIME，防止计时时间太长导致变量累计，因为变量为unsigned int型，为2个字节共65535，超过后就会从0开始计
						KeyState = KEY_LONG_PRESS_CONFIRMED;		//设置成长按标志
						WhichKey_State.KeyName =  KeyNameNow;
					}
					break;

				}
				default:	break;
			}
			//*********************************************************************************************

		}
		else//#################情况1：当前无按键，当前无按键#################
		{//释放按键时，每1ms计数器就执行归零处理
			KeyPressCounterOf1ms = 0;//计数器清零处理
			KeyState = KEY_CHECK;//状态机复位
			KeyStateBefore = KEY_CHECK;
			KeyNameNow = NONE_KEY;
			KeyNameBefore = NONE_KEY;

		}
  }
	else//键值和前一次不同
	{//#################情况3：当前有按键，当前无按键#################
		if((KeyNameBefore != NONE_KEY) && (KeyNameNow == NONE_KEY))
		{
			WhichKey_State.KeyName = KeyNameBefore;//记录释放前的按键值
			if(KeyState == KEY_SHORT_PRESS_CONFIRMED)//确认短按状态退出
			{
				KeyState = KEY_SHORT_RELEASE;
			}
			else if(KeyState == KEY_LONG_PRESS_CONFIRMED)//确认长按的按键状态退出
				KeyState = KEY_LONG_RELEASE;
		}
	}

	WhichKey_State.KeyState = KeyState;

}





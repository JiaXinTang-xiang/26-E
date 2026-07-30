#ifndef __KEY_H__
#define __KEY_H__

typedef unsigned          char uint8_t;
typedef unsigned short     int uint16_t;
typedef unsigned           int uint32_t;


typedef enum
{
	KEY_CHECK = 0,//按键检测状态
	KEY_SHORT_PRESS ,//短按确认状态
	KEY_SHORT_PRESS_CONFIRMED ,//确认短按状态
	KEY_SHORT_RELEASE,//短按抬键状态，前一状态为短按按下，当前状态为抬键
	Key_2TIMES_SHORT_RELEASE,//双击的确认
	KEY_LONG_PRESS_CONFIRMED ,//确认长按状态
	KEY_LONG_RELEASE,//长按抬键状态，前一状态为长按按下，当前状态为抬键
  KEY_RELEASE//按键释放状态，前一状态为抬键，当前状态也为抬键
}KEY_STATE;//按键状态机的状态

typedef enum { NONE_KEY=0, DOWN_KEY, RIGHT_KEY, MID_KEY, LEFT_KEY, UP_KEY, SIDE_KEY }KEY_NAME;//按键键值枚举

typedef enum{ PressNoneKey = 0x00,
							PressKey_Down,PressKey_Right,PressKey_Mid,PressKey_Left,PressKey_Up,PressKey_Side,
							ReleaseKey_Down,ReleaseKey_Right,ReleaseKey_Mid,ReleaseKey_Left,ReleaseKey_Up,ReleaseKey_Side,
							PressAndHoldKey_Down,PressAndHoldKey_Right,PressAndHoldKey_Mid,PressAndHoldKey_Left,PressAndHoldKey_Up,PressAndHoldKey_Side,
							ReleaseHoldedKey_Down,ReleaseHoldedKey_Right,ReleaseHoldedKey_Mid,ReleaseHoldedKey_Left,ReleaseHoldedKey_Up,ReleaseHoldedKey_Side
						}PRESS_KEY;//按键动作枚举


struct KEYSTATE_NAME
{
  KEY_STATE KeyState;
  KEY_NAME KeyName;
};

void JudgeKeyByStateMachine(void) ;//状态机判断按下的哪个按键，以及按键动作，短按、长按、短按抬键、短按抬键等状态来识别双击
void GetKeyStatue(void) ;//PressTheKeyAction的值，放在大循环中处理判断用
char*  FunctionKeyValueState2String(struct KEYSTATE_NAME);

#endif


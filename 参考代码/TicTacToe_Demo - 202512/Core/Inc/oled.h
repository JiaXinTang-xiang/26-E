#ifndef __OLED_H
#define __OLED_H

#include "main.h"
#include "stdlib.h"

//-----------------OLED端口定义----------------

#define OLED_SCL_Clr() HAL_GPIO_WritePin(GPIOB, OLEDSCL_Pin, GPIO_PIN_RESET);//SCL
#define OLED_SCL_Set() HAL_GPIO_WritePin(GPIOB, OLEDSCL_Pin, GPIO_PIN_SET)//SCL

#define OLED_SDA_Clr() HAL_GPIO_WritePin(GPIOB, OLEDSDA_Pin, GPIO_PIN_RESET)//SDA
#define OLED_SDA_Set() HAL_GPIO_WritePin(GPIOB, OLEDSDA_Pin, GPIO_PIN_SET)//SDA

//#define OLED_RES_Clr() GPIO_ResetBits(GPIOA,GPIO_Pin_2)//RES
//#define OLED_RES_Set() GPIO_SetBits(GPIOA,GPIO_Pin_2)


#define OLED_CMD  0	//写命令
#define OLED_DATA 1	//写数据

typedef unsigned char u8;
typedef unsigned int u16;
typedef unsigned long u32;

typedef enum{ MENU_MAIN_WalkAside = 0, MENU_MAIN_MoveTheChess, MENU_MAIN_Z_Axis0,
									MENU_ZAxis,MENU_ZAxis_SHORT_PRESS_UP,MENU_ZAxis_SHORT_PRESS_DOWN,MENU_ZAxis_LONG_PRESS_UP,MENU_ZAxis_LONG_PRESS_DOWN,MENU_ZAxis_SHORT_PRESS_EM_ON,MENU_ZAxis_SHORT_PRESS_EM_OFF
//																								MENU_MAIN_PLAYCHESS2, MENU_MAIN_PUTBACK, MENU_MAIN_CAL, MENU_MAIN_LEDn,
//								MENU_CALI_EXIT,MENU_CALI_XY, MENU_CALI_Z,
//									MENU_CALI_XY0,MENU_CALI_XY1,MENU_CALI_XY2,MENU_CALI_XY3,MENU_CALI_XY4,MENU_CALI_XY5,MENU_CALI_XY_Note,
//
//										MENU_CALI_XY_SHORT_PRESS_LEFT,MENU_CALI_XY_SHORT_PRESS_UP,MENU_CALI_XY_SHORT_PRESS_RIGHT,MENU_CALI_XY_SHOR_PRESST_DOWN,
//										MENU_CALI_XY_LONG_PRESS_LEFT,MENU_CALI_XY_LONG_PRESS_UP,MENU_CALI_XY_LONG_PRESS_RIGHT,MENU_CALI_XY_LONG_PRESS_DOWN,
//											MENU_CALI_XY_CONFIRM_YES,MENU_CALI_XY_CONFIRM_NO,
//									MENU_CALI_Z0,
//										MENU_CALI_Z_SHORT_PRESS_UP,MENU_CALI_Z_SHOR_PRESST_DOWN,MENU_CALI_Z_LONG_PRESS_UP,MENU_CALI_Z_LONG_PRESS_DOWN,MENU_CALI_Z_SHORT_PRESS_EM_ON,MENU_CALI_Z_SHORT_PRESS_EM_OFF,
//										MENU_CALI_Z_CONFIRM_YES,MENU_CALI_Z_CONFIRM_NO

						}MENU_NAME;//显示哪个菜单的变量


void OLED_ClearPoint(u8 x,u8 y);
void OLED_ColorTurn(u8 i);
void OLED_DisplayTurn(u8 i);
void I2C_Start(void);
void I2C_Stop(void);
void I2C_WaitAck(void);
void Send_Byte(u8 dat);
void OLED_WR_Byte(u8 dat,u8 mode);
void OLED_DisPlay_On(void);
void OLED_DisPlay_Off(void);
void OLED_Refresh(void);
void OLED_Clear(void);
void OLED_DrawPoint(u8 x,u8 y,u8 t);
void OLED_DrawLine(u8 x1,u8 y1,u8 x2,u8 y2,u8 mode);
void OLED_DrawCircle(u8 x,u8 y,u8 r);
void OLED_ShowChar(u8 x,u8 y,u8 chr,u8 size1,u8 mode);
void OLED_ShowChar6x8(u8 x,u8 y,u8 chr,u8 mode);
void OLED_ShowString(u8 x,u8 y,u8 *chr,u8 size1,u8 mode);
void OLED_ShowNum(u8 x,u8 y,u32 num,u8 len,u8 size1,u8 mode);
void OLED_ShowChinese(u8 x,u8 y,u8 num,u8 size1,u8 mode);
void OLED_ScrollDisplay(u8 num,u8 space,u8 mode);
void OLED_ShowPicture(u8 x,u8 y,u8 sizex,u8 sizey,u8 BMP[],u8 mode);
void OLED_Init(void);
void OLED_ShowMenu(MENU_NAME MenuNameVar);//显示菜单




#endif


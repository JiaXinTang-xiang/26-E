/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    usart.h
  * @brief   This file contains all the function prototypes for
  *          the usart.c file
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USART_H__
#define __USART_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

extern UART_HandleTypeDef huart1;

/* USER CODE BEGIN Private defines */

#define SERVO_COMMAND_MARKER       0x5A5A
#define COMMAND_PICK_AND_PLACE       0xA1
#define COMMAND_DUAL_SERVO_ANGLE     0xA2
#define STATUS_COMMAND_ACCEPTED      0xB0
#define STATUS_ACTION_COMPLETE       0xB1
#define STATUS_COMMAND_REJECTED      0xB2
#define STATUS_ACTION_FAILED         0xB3

/* USER CODE END Private defines */

void MX_USART1_UART_Init(void);

/* USER CODE BEGIN Prototypes */

uint8_t USART1_RecCommand(void); //串口接收数据处理
void USART1_SendStatus(uint8_t Status);
/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __USART_H__ */


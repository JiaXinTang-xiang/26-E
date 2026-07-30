/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    usart.c
  * @brief   This file provides code for the configuration
  *          of the USART instances.
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
	#include "StepMotor.h"
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "usart.h"

/* USER CODE BEGIN 0 */
uint8_t RecBuf[17]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}; //串口接收数据缓冲区
uint8_t UART1_Rx_flg = 0; //串口接收完成中断标志
uint16_t UART1_Rx_cnt = 17; //串口接收数据个数



uint8_t RecBuf2[12]={0,1,2,3,4,5,7,8,9,10,11}; //缓存12个字节的位置数据
uint16_t PosBuf0[3] = {0,0,0}; //存放棋子的源位置数据
uint16_t PosBuf1[3] = {0,0,0}; //存放棋子的目标位置数据

/* USER CODE END 0 */

UART_HandleTypeDef huart1;

/* USART1 init function */

void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */
		HAL_UART_Receive_IT(&huart1, RecBuf, 17);
  /* USER CODE END USART1_Init 2 */

}

void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(uartHandle->Instance==USART1)
  {
  /* USER CODE BEGIN USART1_MspInit 0 */

  /* USER CODE END USART1_MspInit 0 */
    /* USART1 clock enable */
    __HAL_RCC_USART1_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**USART1 GPIO Configuration
    PA9     ------> USART1_TX
    PA10     ------> USART1_RX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* USART1 interrupt Init */
    HAL_NVIC_SetPriority(USART1_IRQn, 2, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
  /* USER CODE BEGIN USART1_MspInit 1 */

  /* USER CODE END USART1_MspInit 1 */
  }
}

void HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle)
{

  if(uartHandle->Instance==USART1)
  {
  /* USER CODE BEGIN USART1_MspDeInit 0 */

  /* USER CODE END USART1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_USART1_CLK_DISABLE();

    /**USART1 GPIO Configuration
    PA9     ------> USART1_TX
    PA10     ------> USART1_RX
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_9|GPIO_PIN_10);

    /* USART1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(USART1_IRQn);
  /* USER CODE BEGIN USART1_MspDeInit 1 */

  /* USER CODE END USART1_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */
uint8_t USART1_RecCommand(void) //串口接收数据处理
{
	uint8_t temp = 0;

		//		if(HAL_UART_Receive(&huart1,RecBuf,17,100) == HAL_OK) //收到17个字节数据

		//接收数据，要注意第二个字节校验值，且X和Y坐标不能超过步进电机的运动范围
//AA 0F A1 00 C8 00 00 00 00 04 00 00 00 00 00 62 55	//正确数据示例1
//AA 0F A1 00 00 00 64 00 00 00 00 03 80 00 00 49 55	//正确数据示例2
//AA 0F A1 00 96 00 96 00 00 04 01 03 16 00 00 BE 55  //正确数据示例3

//AA 0F A1 01 12 01 AB 00 00 02 34 02 45 00 00 68 55  //错误，校验位错误68应为66
//AA 0F A1 09 12 01 AB 00 00 02 34 02 45 00 00 6E 55  //错误，第一个X坐标超范围
//AA 0F A1 01 12 01 AB 00 00 02 34 09 45 00 00 6D 55  //错误，第二个Y坐标超范围



			HAL_UART_Transmit(&huart1,RecBuf,17,10); //显示接收的数据，调试时可以打开

			temp = RecBuf[1];
			for(uint8_t i = 2;i<=14;i++) //计算校验值
      {
				temp = temp ^ RecBuf[i];
      }

			HAL_UART_Transmit(&huart1,&temp,1,10); //显示计算的校验值，调试时可以打开

			if(RecBuf[0] == 0xAA && RecBuf[16] == 0x55 && RecBuf[15] == temp)
			{

				PosBuf0[0] = RecBuf[3]<<8 | RecBuf[4]; //3个16位数据，源位置数据
				PosBuf0[1] = RecBuf[5]<<8 | RecBuf[6];
				PosBuf0[2] = RecBuf[7]<<8 | RecBuf[8];
				PosBuf1[0] = RecBuf[9]<<8 | RecBuf[10]; //3个16位数据，目标位置数据
				PosBuf1[1] = RecBuf[11]<<8 | RecBuf[12];
				PosBuf1[2] = RecBuf[13]<<8 | RecBuf[14];
				if(PosBuf0[0] <= MAXPosX && PosBuf0[1] <= MAXPosY && PosBuf1[0] <= MAXPosX && PosBuf1[1] <= MAXPosY) //判断坐标范围
				{
					//将6个16位数据转换为12个字节，用于发送验证
					RecBuf2[0] = PosBuf0[0]>>8; //取高8位
					RecBuf2[1] = PosBuf0[0];    //取低8位
					RecBuf2[2] = PosBuf0[1]>>8;
					RecBuf2[3] = PosBuf0[1];
					RecBuf2[4] = PosBuf0[2]>>8;
					RecBuf2[5] = PosBuf0[2];
					RecBuf2[6] = PosBuf1[0]>>8; //取高8位
					RecBuf2[7] = PosBuf1[0];    //取低8位
					RecBuf2[8] = PosBuf1[1]>>8;
					RecBuf2[9] = PosBuf1[1];
					RecBuf2[10] = PosBuf1[2]>>8;
					RecBuf2[11] = PosBuf1[2];
					HAL_UART_Transmit(&huart1,RecBuf2,12,10); //显示获取的数据值，调试时可以打开
					return(1);
				}
			}


			Beep(200); //蜂鸣器报警，数据校验失败，原因是校验值错误或X/Y坐标超出步进电机运动范围
			return(0);


}

/* USER CODE END 1 */

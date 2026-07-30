/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
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
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define LED2_Pin GPIO_PIN_13
#define LED2_GPIO_Port GPIOC
#define PA6_LSZ_Pin GPIO_PIN_6
#define PA6_LSZ_GPIO_Port GPIOA
#define LEDn_Pin GPIO_PIN_1
#define LEDn_GPIO_Port GPIOB
#define EM_Pin GPIO_PIN_12
#define EM_GPIO_Port GPIOB
#define STEPZ_EN_Pin GPIO_PIN_13
#define STEPZ_EN_GPIO_Port GPIOB
#define PB14_LSY_Pin GPIO_PIN_14
#define PB14_LSY_GPIO_Port GPIOB
#define PB15_LSX_Pin GPIO_PIN_15
#define PB15_LSX_GPIO_Port GPIOB
#define TIM_CH1_Pin GPIO_PIN_8
#define TIM_CH1_GPIO_Port GPIOA
#define STEPX_DIR_Pin GPIO_PIN_11
#define STEPX_DIR_GPIO_Port GPIOA
#define STEPZ_DIR_Pin GPIO_PIN_12
#define STEPZ_DIR_GPIO_Port GPIOA
#define SWDIO_Pin GPIO_PIN_13
#define SWDIO_GPIO_Port GPIOA
#define SWCLK_Pin GPIO_PIN_14
#define SWCLK_GPIO_Port GPIOA
#define Beep_Pin GPIO_PIN_3
#define Beep_GPIO_Port GPIOB
#define STEPX_EN_Pin GPIO_PIN_4
#define STEPX_EN_GPIO_Port GPIOB
#define STEPY_DIR_Pin GPIO_PIN_5
#define STEPY_DIR_GPIO_Port GPIOB
#define OLEDSCL_Pin GPIO_PIN_6
#define OLEDSCL_GPIO_Port GPIOB
#define OLEDSDA_Pin GPIO_PIN_7
#define OLEDSDA_GPIO_Port GPIOB
#define LED1_Pin GPIO_PIN_8
#define LED1_GPIO_Port GPIOB
#define STEPY_EN_Pin GPIO_PIN_9
#define STEPY_EN_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */
void Beep(uint16_t BeepTime);
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

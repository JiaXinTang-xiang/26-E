#ifndef __SERVO_H
#define __SERVO_H

#include "main.h"

#define SERVO_PWM_PERIOD          20000U
#define SERVO_MIN_PULSE             500U
#define SERVO_MAX_PULSE            2500U
#define SERVO_MAX_ANGLE              270U

#define SERVO_HOME_ANGLE             135U
#define SERVO_PICKED_ANGLE            90U
#define SERVO_MOVE_DELAY             400U

void SERVO_Init(void);
void SERVO_SetAngle(uint16_t Angle);
void SERVO_MoveToAngle(uint16_t Angle);

#endif

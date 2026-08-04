#ifndef __SERVO_H
#define __SERVO_H

#include "main.h"

#define SERVO_PWM_PERIOD          20000U
#define SERVO_MIN_PULSE             500U
#define SERVO_MAX_PULSE            2500U
#define SERVO_MAX_ANGLE              270U

#define SERVO_HOME_ANGLE             135U
#define SERVO_PICKED_ANGLE            90U

/*
 * 270-degree servo movement time depends on the commanded angle change.
 * Keep a small settling delay after the estimated travel time so the Z axis
 * never descends while the suction head is still rotating.
 */
#define SERVO_MOVE_MS_PER_DEGREE        5U
#define SERVO_SETTLE_DELAY            250U
#define SERVO_INITIAL_MOVE_DELAY      1000U
#define SERVO_MAX_MOVE_DELAY          1800U

void SERVO_Init(void);
void SERVO_SetAngle(uint16_t Angle);
void SERVO_MoveToAngle(uint16_t Angle);

#endif

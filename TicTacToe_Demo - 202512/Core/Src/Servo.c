#include "Servo.h"
#include "tim.h"

static uint16_t ServoCurrentAngle = 0xFFFFU;

void SERVO_Init(void)
{
	HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_3);
	SERVO_MoveToAngle(SERVO_HOME_ANGLE);
}

void SERVO_SetAngle(uint16_t Angle)
{
	uint32_t Pulse;

	if(Angle > SERVO_MAX_ANGLE)
	{
		Angle = SERVO_MAX_ANGLE;
	}

	Pulse = SERVO_MIN_PULSE
			+ (uint32_t)Angle * (SERVO_MAX_PULSE - SERVO_MIN_PULSE)
			/ SERVO_MAX_ANGLE;

	__HAL_TIM_SET_COMPARE(&htim4, TIM_CHANNEL_3, Pulse);
	ServoCurrentAngle = Angle;
}

void SERVO_MoveToAngle(uint16_t Angle)
{
	if(Angle > SERVO_MAX_ANGLE)
	{
		Angle = SERVO_MAX_ANGLE;
	}

	if(ServoCurrentAngle == Angle)
	{
		return;
	}

	SERVO_SetAngle(Angle);
	HAL_Delay(SERVO_MOVE_DELAY);
}

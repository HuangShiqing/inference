#include <wiringPi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include "step_motor.h"
#include "network.h"
#include "my_common.h"

#define CLOCKWISE 1
#define COUNTER_CLOCKWISE 2


extern pthread_mutex_t mutex3;
extern detection* g_dets;

void *step_motor_thread(void *arg)
{
    detection* det = calloc(1, sizeof(detection));
    int* pins = (int *)arg;
    delayMS(3000);
    printf("start following\r\n");
    while(1)
    {
        pthread_mutex_lock(&mutex3);
        memcpy(det, g_dets, sizeof(detection));
		pthread_mutex_unlock(&mutex3);

        int delta = (det->bbox.x- 0.5)*WIDTH;

        if(det->max_prob > 0.8)
        {
            if( delta > 10)//右侧
            {
                // printf("bbox.x*WIDTH: %f, in right\r\n",det->bbox.x*WIDTH);
                // for(int i=0;i<10;i++)
                rotate(pins, COUNTER_CLOCKWISE);
            }
            else if(delta < -10 && delta > -(WIDTH/2))//左侧
            {
                // printf("bbox.x*WIDTH: %f, in left\r\n",det->bbox.x*WIDTH);
                // for(int i=0;i<10;i++)
                rotate(pins, CLOCKWISE);
            }
            else//死区
            {
                ;
                // printf("in middle\r\n");
            }                
        }
        // delayMS(50);

    }
}

int pins[4];

int step_motor_init()
{
    // if (argc < 4)
    // {
    //     printf("Usage example: ./motor 0 1 2 3 \n");
    //     return 1;
    // }
    // /* number of the pins which connected to the stepper motor driver board */
    int pinA = 0;//atoi(argv[1]);
    int pinB = 1;//atoi(argv[2]);
    int pinC = 2;//atoi(argv[3]);
    int pinD = 3;//atoi(argv[4]);

    pins[0] = pinA;
    pins[1] = pinB;
    pins[2] = pinC;
    pins[3] = pinD;
    // pins[4] = {pinA, pinB, pinC, pinD};

    if (-1 == wiringPiSetup())
    {
        printf("Setup wiringPi failed!");
        return 1;
    }

    /* set mode to output */
    pinMode(pinA, OUTPUT);
    pinMode(pinB, OUTPUT);
    pinMode(pinC, OUTPUT);
    pinMode(pinD, OUTPUT);

    delayMS(50); // wait for a stable status

    pthread_t step_motor_tid;
    int r = pthread_create(&step_motor_tid, 0, step_motor_thread, (void *)&(pins));
    if (r != 0)
        printf("step_motor Thread creation failed");

    return 0;
}

/* Suspend execution for x milliseconds intervals.
 *  *  @param ms Milliseconds to sleep.
 *   */
void delayMS(int x)
{
    usleep(x * 1000);
}

/* Rotate the motor.
 *  *  @param pins     A pointer which points to the pins number array.
 *  *  @param direction  CLOCKWISE for clockwise rotation, COUNTER_CLOCKWISE for counter clockwise rotation.
 *  */
void rotate(int *pins, int direction)
{
    for (int i = 0; i < 4; i++)
    {
        if (CLOCKWISE == direction)
        {
            for (int j = 0; j < 4; j++)
            {
                if (j == i)
                {
                    digitalWrite(pins[3 - j], 1); // output a high level
                }
                else
                {
                    digitalWrite(pins[3 - j], 0); // output a low level
                }
            }
        }
        else if (COUNTER_CLOCKWISE == direction)
        {
            for (int j = 0; j < 4; j++)
            {
                if (j == i)
                {
                    digitalWrite(pins[j], 1); // output a high level
                }
                else
                {
                    digitalWrite(pins[j], 0); // output a low level
                }
            }
        }
        delayMS(4);
    }
}
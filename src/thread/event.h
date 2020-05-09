#ifndef EVENT_H
#define EVENT_H

#include <pthread.h>

typedef struct
{
    pthread_cond_t mID;
    pthread_mutex_t mQueryMutex;
    int mAutoReset;
    int mQuery;
} event_parameter;

void event_init(event_parameter *event_par, int autoreset);
void event_wake(event_parameter *event_par);
int event_wait(event_parameter *event_par);
#endif
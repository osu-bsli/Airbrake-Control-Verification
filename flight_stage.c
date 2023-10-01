#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define FILE_NAME "data/2021-03-20-serial-4980-flight-0003.csv" // 2021-03-20-serial-4980-flight-0003.csv 

float time;
float accel;
float alt;
float velo;

void step(FILE *fp) {
    char row[1000];
    char *pt;
    int i=0;
    fgets(row, 1000, fp);
    pt = strtok (row,",");
    for (i; pt != NULL; i++) {
        float a = atof(pt);
        switch (i) {
            case 4:
                time = a;
                break;
            case 7:
                accel = a;
                break;
            case 9:
                alt = a;
                break;
            case 11:
                velo = a;
                break;
        }
        pt = strtok (NULL, ",");
    }
}

int main() {
    // Setup
    FILE *fp;
    int apogee = 0;
    char row[1000];
    fp = fopen(FILE_NAME,"r");
    fgets(row, 1000, fp);

    // adding variables here so we don't keep looping over them
    int waitTillAlt = 200;
    float lastVelo = 0;
    int a = 0;
    int b = 0;
    do
    {
        step(fp);

        //TODO: are we at apogee?
        // we are going to test when the velocity switches from positive to negative
        if (alt < waitTillAlt) {
            // do nothing
        } else {
            // do stuff
            if (velo < 0) {
                a++;
            } else {
                b++;
            }
            if (b > 10) {
                a = 0;
                b = 0;
            }
            if (a > 10) {
                apogee = 1;
                printf("%f", alt);
                break;
            }
        }
        printf("Time: %f \t\t\t Apogee? %d\n", time, apogee);
    } while (feof(fp) != 1);
    return 0;
}
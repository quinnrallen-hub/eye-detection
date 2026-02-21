// Quick hardware test — turns everything on

#define BUZZER_PIN   25
#define LED_PIN      12
#define ONBOARD_LED   2

#define BUZZ_FREQ     2000  // KY-006 optimal range: 1500-2500 Hz
#define BUZZ_RES      10    // 10-bit resolution (0-1023)
#define BUZZ_DUTY     512   // 50% duty cycle = loudest

void setup() {
    pinMode(LED_PIN,     OUTPUT);
    pinMode(ONBOARD_LED, OUTPUT);

    // Use LEDC (proper ESP32 PWM) instead of tone()
    ledcAttach(BUZZER_PIN, BUZZ_FREQ, BUZZ_RES);
    ledcWrite(BUZZER_PIN, BUZZ_DUTY);

    digitalWrite(LED_PIN,     HIGH);
    digitalWrite(ONBOARD_LED, HIGH);
}

void loop() {}

// ESP32 Alert Controller
// KY-006 passive buzzer on GPIO 14, LED on GPIO 12
// gui.py hits /alert/on and /alert/off over WiFi

#include <WiFi.h>
#include <WebServer.h>

// === CHANGE THESE ===
const char* ssid     = "bee0126";
const char* password = "";

#define BUZZER_PIN   25
#define BUZZ_FREQ     2000  // KY-006 optimal range: 1500-2500 Hz
#define BUZZ_RES      10    // 10-bit resolution
#define BUZZ_DUTY     512   // 50% = loudest
#define LED_PIN      12
#define ONBOARD_LED   2

WebServer server(80);

bool alertActive = false;

void handleAlertOn() {
    alertActive = true;
    ledcWrite(BUZZER_PIN, BUZZ_DUTY);
    digitalWrite(LED_PIN,     HIGH);
    digitalWrite(ONBOARD_LED, HIGH);
    server.send(200, "text/plain", "on");
}

void handleAlertOff() {
    alertActive = false;
    ledcWrite(BUZZER_PIN, 0);
    digitalWrite(LED_PIN,     LOW);
    digitalWrite(ONBOARD_LED, LOW);
    server.send(200, "text/plain", "off");
}

void handleStatus() {
    server.send(200, "text/plain", alertActive ? "on" : "off");
}

void safeState() {
    alertActive = false;
    ledcWrite(BUZZER_PIN, 0);
    digitalWrite(LED_PIN, LOW);
}

bool connectWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect(true);
    delay(100);
    WiFi.begin(ssid, password);
    Serial.print("Connecting");

    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - start > 15000) {
            Serial.println("\nWiFi timed out.");
            digitalWrite(ONBOARD_LED, LOW); // LED off
            return false;
        }
        delay(300);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("Connected! Alert server at http://");
    Serial.println(WiFi.localIP());

    return true;
}

void setup() {
    Serial.begin(115200);

    pinMode(LED_PIN,     OUTPUT);
    pinMode(ONBOARD_LED, OUTPUT);

    ledcAttach(BUZZER_PIN, BUZZ_FREQ, BUZZ_RES);

    safeState();

    if (!connectWiFi()) {
        Serial.println("WiFi failed — halting");
        while (true) delay(1000);
    }

    digitalWrite(ONBOARD_LED, HIGH); // On = WiFi connected

    server.on("/alert/on",  handleAlertOn);
    server.on("/alert/off", handleAlertOff);
    server.on("/status",    handleStatus);
    server.begin();
}

// Track last reconnect attempt to avoid hammering
unsigned long lastReconnectAttempt = 0;

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        unsigned long now = millis();
        if (now - lastReconnectAttempt > 10000) {
            lastReconnectAttempt = now;
            Serial.println("WiFi lost — reconnecting...");
            safeState(); // ensure buzzer off during reconnect
            if (connectWiFi()) {
                Serial.println("Reconnected.");
            }
        }
        return; // don't call server.handleClient() while disconnected
    }
    server.handleClient();
}

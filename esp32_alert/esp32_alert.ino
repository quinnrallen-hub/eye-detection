// ESP32 Alert Controller
// Buzzer on GPIO 25, LED on GPIO 26
// gui.py hits /alert/on and /alert/off over WiFi

#include <WiFi.h>
#include <WebServer.h>

// === CHANGE THESE ===
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

#define BUZZER_PIN 25
#define LED_PIN    26

WebServer server(80);

void handleAlertOn() {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    server.send(200, "text/plain", "on");
}

void handleAlertOff() {
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    server.send(200, "text/plain", "off");
}

void handleStatus() {
    bool on = digitalRead(LED_PIN);
    server.send(200, "text/plain", on ? "on" : "off");
}

void setup() {
    Serial.begin(115200);

    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN, LOW);

    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! Alert server at http://");
    Serial.println(WiFi.localIP());

    server.on("/alert/on",  handleAlertOn);
    server.on("/alert/off", handleAlertOff);
    server.on("/status",    handleStatus);
    server.begin();
}

void loop() {
    server.handleClient();
}

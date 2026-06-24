// AppHeader.qml — gradient header bar with title, lang selector, ready badge
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Item {
    height: 64

    // gradient background
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "#1a1040" }
            GradientStop { position: 1.0; color: "#0d1b2a" }
        }
    }
    // bottom border
    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width; height: 1; color: "#3a3f5c"
    }

    RowLayout {
        anchors { fill: parent; leftMargin: 20; rightMargin: 20 }
        spacing: 12

        // ── title block ───────────────────────────────────────────
        Column {
            spacing: 0
            Label {
                id: titleLbl
                text: i18n.t("app_title")
                font.pixelSize: 16; font.bold: true
                color: "#80cbc4"
            }
            Label {
                id: subtitleLbl
                text: i18n.t("app_subtitle")
                font.pixelSize: 10; color: "#607d8b"
            }
        }

        Item { Layout.fillWidth: true }

        // ── language selector ─────────────────────────────────────
        RowLayout {
            spacing: 6
            Label { text: "🌐"; font.pixelSize: 14; color: "#b0bec5" }
            ComboBox {
                id: langCombo
                model: i18n.displayNames
                currentIndex: {
                    var codes = i18n.languageCodes
                    for (var i = 0; i < codes.length; i++)
                        if (codes[i] === i18n.currentLanguage) return i
                    return 0
                }
                implicitWidth: 130
                Material.accent: "#80cbc4"
                onActivated: i18n.setLanguage(i18n.languageCodes[currentIndex])
            }
        }

        // ── system-ready badge ────────────────────────────────────
        Rectangle {
            radius: 4
            implicitWidth:  readyLbl.implicitWidth + 20
            implicitHeight: 24
            color: status.mfaInstalled ? "#1b5e20" : "#b71c1c"
            Behavior on color { ColorAnimation { duration: 200 } }

            Label {
                id: readyLbl
                anchors.centerIn: parent
                text: status.mfaInstalled
                    ? i18n.t("system_ready")
                    : i18n.t("system_not_ready")
                font.pixelSize: 11; font.bold: true
                color: "#ffffff"
            }
        }
    }

    // retranslate on lang change
    Connections {
        target: i18n
        function onLanguageChanged() {
            titleLbl.text    = i18n.t("app_title")
            subtitleLbl.text = i18n.t("app_subtitle")
            readyLbl.text    = status.mfaInstalled
                ? i18n.t("system_ready")
                : i18n.t("system_not_ready")
        }
    }
}

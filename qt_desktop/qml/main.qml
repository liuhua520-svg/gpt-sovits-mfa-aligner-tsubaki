// main.qml — root window
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: root
    title: "SVS Lab Aligner"
    width: 940
    height: 880
    minimumWidth: 720
    minimumHeight: 600
    visible: true

    // ── Material dark theme ───────────────────────────────────────
    Material.theme:      Material.Dark
    Material.accent:     "#80cbc4"
    Material.primary:    "#1e2030"
    Material.background: "#141624"
    Material.foreground: "#eceff1"

    background: Rectangle { color: "#141624" }

    // ── main layout ───────────────────────────────────────────────
    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        AppHeader  { Layout.fillWidth: true }
        BackendBar { Layout.fillWidth: true }

        ScrollView {
            Layout.fillWidth:  true
            Layout.fillHeight: true
            clip: true
            contentWidth: availableWidth

            ColumnLayout {
                width: parent.width
                spacing: 16

                Item { Layout.preferredHeight: 4 }

                WarningBanner {
                    id: mfaWarn
                    Layout.fillWidth:    true
                    Layout.leftMargin:   20
                    Layout.rightMargin:  20
                    kind:    "error"
                    visible: false
                }

                ProcessorWidget {
                    id: processor
                    Layout.fillWidth:   true
                    Layout.leftMargin:  20
                    Layout.rightMargin: 20
                }

                SystemStatusWidget {
                    Layout.fillWidth:   true
                    Layout.leftMargin:  20
                    Layout.rightMargin: 20
                }

                WarningBanner {
                    id: modelsWarn
                    Layout.fillWidth:   true
                    Layout.leftMargin:  20
                    Layout.rightMargin: 20
                    kind:    "warning"
                    visible: false
                }

                Item { Layout.preferredHeight: 20 }
            }
        }

        AppFooter { Layout.fillWidth: true }
    }

    // ── status bar ────────────────────────────────────────────────
    footer: ToolBar {
        height: 22
        background: Rectangle { color: "#0d1117" }
        Label {
            id: statusMsg
            anchors.verticalCenter: parent.verticalCenter
            leftPadding: 12
            text: "SVS Lab Aligner  |  " + api.baseUrl
            color: "#607d8b"
            font.pixelSize: 10
        }
    }

    // ── API event handlers ────────────────────────────────────────
    Connections {
        target: api

        function onStatusFetched() {
            statusMsg.text = "SVS Lab Aligner  |  " + api.baseUrl
                + "  |  " + i18n.t("msg_status_refreshed")
            _updateWarnings()
        }
        function onStatusFailed(err) {
            statusMsg.text = i18n.t("msg_backend_unreachable") + ":  " + err
        }
        function onModelDownloaded(lang) {
            statusMsg.text = i18n.tf("msg_model_ok", {"lang": lang})
        }
        function onModelFailed(lang, err) {
            statusMsg.text = i18n.tf("msg_model_fail", {"error": err})
        }
    }

    function _updateWarnings() {
        if (!status.mfaInstalled) {
            mfaWarn.messageText =
                "<b>" + i18n.t("warn_mfa_not_installed") + "</b><br>"
                + "<code>" + i18n.t("warn_mfa_install_cmd") + "</code><br>"
                + i18n.t("warn_install_hint")
            mfaWarn.visible  = true
            modelsWarn.visible = false
        } else {
            mfaWarn.visible = false
            var m = status.models
            var allOk = m["cmn"] && m["eng"] && m["jpn"] && m["kor"] && m["yue"]
            if (!allOk) {
                modelsWarn.messageText =
                    "<b>" + i18n.t("warn_not_ready_title") + "</b><br>"
                    + i18n.t("warn_not_ready_msg")
                modelsWarn.visible = true
            } else {
                modelsWarn.visible = false
            }
        }
    }

    // ── initial status check ──────────────────────────────────────
    Timer {
        interval: 500; running: true; repeat: false
        onTriggered: api.checkStatus()
    }
}

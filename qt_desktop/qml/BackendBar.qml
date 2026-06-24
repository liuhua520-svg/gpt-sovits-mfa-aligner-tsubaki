// BackendBar.qml — backend URL input bar
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Item {
    height: 40

    Rectangle { anchors.fill: parent; color: "#141624" }
    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: "#3a3f5c" }

    RowLayout {
        anchors { fill: parent; leftMargin: 16; rightMargin: 16 }
        spacing: 8

        Label {
            id: urlLbl
            text: i18n.t("backend_url_label")
            font.pixelSize: 11; color: "#607d8b"
        }

        TextField {
            id: urlField
            text: api.baseUrl
            implicitWidth: 240
            font.pixelSize: 11
            color: "#eceff1"
            placeholderText: "http://127.0.0.1:5000"
            background: Rectangle {
                color: "#1e2030"; radius: 3
                border.color: urlField.activeFocus ? "#80cbc4" : "#3a3f5c"
                border.width: 1
            }
            onAccepted: _apply()
        }

        Button {
            id: connBtn
            text: i18n.t("connect")
            implicitWidth: 80
            Material.background: "#00695c"
            Material.foreground: "#ffffff"
            onClicked: _apply()
        }

        Item { Layout.fillWidth: true }
    }

    function _apply() {
        var u = urlField.text.trim().replace(/\/+$/, "")
        if (u) api.setBaseUrl(u)
    }

    Connections {
        target: i18n
        function onLanguageChanged() {
            urlLbl.text  = i18n.t("backend_url_label")
            connBtn.text = i18n.t("connect")
        }
    }
    Connections {
        target: api
        function onBaseUrlChanged(url) { urlField.text = url }
    }
}

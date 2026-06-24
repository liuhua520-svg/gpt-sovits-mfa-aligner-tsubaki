// AppFooter.qml — bottom footer bar
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    height: 36

    Rectangle { anchors.fill: parent; color: "#0d1117" }
    Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: "#21262d" }

    RowLayout {
        anchors { fill: parent; leftMargin: 20; rightMargin: 12 }

        Label {
            id: footerLbl
            text: i18n.t("footer_text")
            font.pixelSize: 10; color: "#484f58"
        }
        Item { Layout.fillWidth: true }
        Label {
            text: "📚 GitHub"
            font.pixelSize: 10
            color: ghArea.containsMouse ? "#79c0ff" : "#58a6ff"
            MouseArea {
                id: ghArea
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: Qt.openUrlExternally(
                    "https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner-tsubaki"
                )
            }
        }
    }

    Connections {
        target: i18n
        function onLanguageChanged() { footerLbl.text = i18n.t("footer_text") }
    }
}

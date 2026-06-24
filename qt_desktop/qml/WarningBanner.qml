// WarningBanner.qml — error / warning banner
import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    property string kind:        "error"   // "error" | "warning"
    property string messageText: ""

    readonly property color _bg:  kind === "error" ? "#4a1010" : "#3a2800"
    readonly property color _fg:  kind === "error" ? "#ef9a9a" : "#ffe082"
    readonly property color _brd: kind === "error" ? "#b71c1c" : "#f57f17"

    radius: 6
    color:  _bg
    border.color: _brd
    border.width: 1
    implicitHeight: msg.implicitHeight + 20

    Label {
        id: msg
        anchors { fill: parent; margins: 10 }
        text: messageText
        wrapMode: Text.WordWrap
        color: root._fg
        textFormat: Text.RichText
        font.pixelSize: 12
    }
}

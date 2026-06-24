// StatusBadge.qml — coloured pill label (green = ok, red = not ok)
import QtQuick
import QtQuick.Controls

Rectangle {
    property bool   ok:    false
    property string label: ""

    radius: 4
    color:  ok ? "#4caf50" : "#f44336"
    implicitWidth:  lbl.implicitWidth + 16
    implicitHeight: 22

    Behavior on color { ColorAnimation { duration: 150 } }

    Label {
        id: lbl
        anchors.centerIn: parent
        text:  parent.label
        color: "#ffffff"
        font.pixelSize: 11
        font.bold: true
    }
}

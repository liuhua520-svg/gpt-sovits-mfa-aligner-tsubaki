// FormRow.qml — 110 px label on the left, content fills the right
// Usage:
//   FormRow {
//       labelText: "Audio File"
//       FileDropZone { width: parent.width }
//   }
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    property string labelText: ""

    // Items declared inside FormRow { } go into rightSide as children
    default property alias content: rightSide.data

    width:  parent ? parent.width : 200
    height: Math.max(lbl.implicitHeight, rightSide.childrenRect.height) + 2

    Label {
        id: lbl
        text:  root.labelText
        width: 110
        color: "#b0bec5"
        font.pixelSize: 12
        horizontalAlignment: Text.AlignRight
        wrapMode: Text.WordWrap
        anchors { left: parent.left; top: parent.top; topMargin: 2 }
    }

    Item {
        id: rightSide
        anchors {
            left:       lbl.right
            leftMargin: 12
            right:      parent.right
            top:        parent.top
        }
        height: childrenRect.height || 4
    }
}

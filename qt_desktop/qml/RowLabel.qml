// RowLabel.qml — 140 px label on the left, content fills the right.
// Used inside CollapsibleSection for the advanced-settings grid.
// Usage:
//   RowLabel {
//       labelText: "BPM"
//       SpinBox { from: 20; to: 300; value: 120 }
//   }
import QtQuick
import QtQuick.Controls

Item {
    id: root
    property string labelText: ""

    default property alias content: rightSide.data

    width:  parent ? parent.width : 200
    height: Math.max(lbl.implicitHeight, rightSide.childrenRect.height) + 2

    Label {
        id: lbl
        text:  root.labelText
        width: 140
        color: "#b0bec5"
        font.pixelSize: 11
        horizontalAlignment: Text.AlignRight
        wrapMode: Text.WordWrap
        anchors { left: parent.left; top: parent.top; topMargin: 2 }
    }

    Item {
        id: rightSide
        anchors {
            left:       lbl.right
            leftMargin: 8
            right:      parent.right
            top:        parent.top
        }
        height: childrenRect.height || 4
    }
}

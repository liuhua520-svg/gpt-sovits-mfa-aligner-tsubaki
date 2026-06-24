// CollapsibleSection.qml — expandable panel with arrow toggle
// Usage:
//   CollapsibleSection {
//       title: "Advanced Settings"
//       RowLabel { labelText: "BPM"; SpinBox { } }
//       RowLabel { labelText: "Key";  ComboBox { } }
//   }
import QtQuick
import QtQuick.Controls

Column {
    id: root
    property string title: ""

    // Items declared inside CollapsibleSection { } flow into contentColumn
    default property alias content: contentColumn.data

    spacing: 0

    // ── toggle button ─────────────────────────────────────────────
    Button {
        id: toggleBtn
        width: parent.width
        flat: true; checkable: true; checked: false
        padding: 0; leftPadding: 2

        contentItem: Row {
            spacing: 6
            Label { text: toggleBtn.checked ? "▼" : "▶"; color: "#80cbc4"; font.pixelSize: 11 }
            Label { text: root.title; color: "#80cbc4"; font.pixelSize: 12; font.bold: true }
        }
        background: Item {}
    }

    // ── collapsible content area ──────────────────────────────────
    Rectangle {
        id: body
        width: parent.width
        height: toggleBtn.checked ? contentColumn.implicitHeight + 20 : 0
        visible: height > 0
        clip:    true
        color:  "#1a1d2e"
        radius:  6
        border.color: "#3a3f5c"; border.width: 1

        Behavior on height {
            NumberAnimation { duration: 160; easing.type: Easing.InOutQuad }
        }

        Column {
            id: contentColumn
            anchors { left: parent.left; right: parent.right; top: parent.top }
            anchors.margins: 12
            spacing: 10
        }
    }
}

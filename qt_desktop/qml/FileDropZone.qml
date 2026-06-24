// FileDropZone.qml — click-to-browse + drag-and-drop file picker
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Rectangle {
    id: root

    // ── API ───────────────────────────────────────────────────────
    property var    acceptExts: []     // e.g. ["wav","mp3","flac"]
    property string hint:       ""
    property string tipText:    ""
    property string filePath:   ""     // bound two-way by parent

    signal fileSelected(string path)
    signal cleared()

    // ── appearance ────────────────────────────────────────────────
    height: 100; radius: 8
    color:  filePath !== "" ? "#1e2a1e" : "#1a1d2e"
    border.width: 2
    border.color: dropArea.containsDrag ? "#80cbc4"
                : filePath !== ""       ? "#4caf50"
                :                         "#3a3f5c"

    Behavior on color        { ColorAnimation { duration: 150 } }
    Behavior on border.color { ColorAnimation { duration: 150 } }

    // ── drag-and-drop ─────────────────────────────────────────────
    DropArea {
        id: dropArea
        anchors.fill: parent
        keys: ["text/uri-list"]
        onDropped: (drop) => {
            if (drop.hasUrls && drop.urls.length > 0)
                _tryAccept(_urlToPath(drop.urls[0].toString()))
        }
    }

    // ── content ───────────────────────────────────────────────────
    ColumnLayout {
        anchors { fill: parent; margins: 8 }
        spacing: 3

        Label {
            Layout.alignment: Qt.AlignHCenter
            text: root.filePath !== "" ? "✅" : "📂"
            font.pixelSize: 20
        }
        Label {
            Layout.alignment: Qt.AlignHCenter
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            text: root.filePath !== ""
                ? _fileName(root.filePath) + "  (" + api.fileSizeStr(root.filePath) + ")"
                : root.hint
            color: root.filePath !== "" ? "#4caf50" : "#b0bec5"
            font.pixelSize: 11
            wrapMode: Text.WordWrap
        }
        Label {
            Layout.alignment: Qt.AlignHCenter
            text: root.tipText
            color: "#607d8b"; font.pixelSize: 10
            visible: root.tipText !== "" && root.filePath === ""
        }
        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: 6
            Button {
                id: browseBtn
                text: i18n.t("browse")
                implicitWidth: 90; font.pixelSize: 11
                onClicked: fileDialog.open()
            }
            Button {
                text: "✕"; implicitWidth: 36; font.pixelSize: 11
                visible: root.filePath !== ""
                onClicked: _clearFile()
            }
        }
    }

    // click anywhere on the zone also opens the dialog
    MouseArea {
        anchors.fill: parent; z: -1
        cursorShape: Qt.PointingHandCursor
        onClicked: fileDialog.open()
    }

    // ── file dialog ───────────────────────────────────────────────
    FileDialog {
        id: fileDialog
        title: root.hint
        nameFilters: ["Files (*." + root.acceptExts.join(" *.") + ")"]
        onAccepted: _tryAccept(_urlToPath(selectedFile.toString()))
    }

    // ── helpers ───────────────────────────────────────────────────
    function _tryAccept(path) {
        if (path === "") return
        var ext = path.split(".").pop().toLowerCase()
        if (root.acceptExts.indexOf(ext) < 0) return
        root.filePath = path
        root.fileSelected(path)
    }
    function _clearFile() {
        root.filePath = ""
        root.cleared()
    }
    function _fileName(p) {
        return p.split(/[/\\]/).pop()
    }
    function _urlToPath(url) {
        // file:///C:/... → C:/...   |   file:///home/... → /home/...
        var s = url.toString()
        s = s.replace(/^file:\/\/\/([A-Za-z]:)/, "$1")  // Windows
        s = s.replace(/^file:\/\//, "")                  // Unix
        return s
    }

    // update i18n on browse button
    Connections {
        target: i18n
        function onLanguageChanged() { browseBtn.text = i18n.t("browse") }
    }
}

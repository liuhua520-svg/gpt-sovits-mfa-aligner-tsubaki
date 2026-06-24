// ProcessorWidget.qml — main processing form
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

Rectangle {
    id: root
    radius: 10; color: "#1e2030"
    border.color: "#3a3f5c"; border.width: 1
    implicitHeight: mainCol.implicitHeight + 36

    // ══════════════════════════════════════════════════════════════
    //  FORM STATE
    // ══════════════════════════════════════════════════════════════
    property string mode:          "mfa-only"   // mfa-only | full | project-only
    property string backend:       "mfa"
    property string alignerDevice: "auto"
    property string language:      "cmn"
    property string outputFormat:  "sv"
    property string phonemeMode:   "none"
    property string f0Method:      "dio"
    property string f0Device:      "auto"
    property string crepeModel:    "full"
    property string precision:     "single"

    property string audioPath:   ""
    property string labMidiPath: ""
    property string trackName:   "Project"
    property string inputText:   ""

    property int    bpm:            120
    property int    basePitch:      60
    property bool   autoNotePitch:  true
    property bool   exportPitchLine:true
    property bool   f0Smooth:       true
    property int    f0SmoothWindow: 5
    property int    f0Floor:        71
    property int    f0Ceil:         800
    property bool   midiLocked:     false   // true when MIDI loaded

    // ── job state ─────────────────────────────────────────────────
    property bool   processing:     false
    property int    progressValue:  0
    property var    jobResult:      null
    property string errorText:      ""

    // ── derived visibility flags ───────────────────────────────────
    readonly property bool isProjectOnly: mode === "project-only"
    readonly property bool isMfaOnly:     mode === "mfa-only"
    readonly property bool isFull:        mode === "full"
    readonly property bool isMfaBck:      backend === "mfa"
    readonly property bool showDevice:    !isMfaBck && !isProjectOnly
    readonly property bool textOptional:  backend === "whisperx" || backend === "qwen3_asr"

    // ── start enabled check ────────────────────────────────────────
    readonly property bool canStart: {
        if (processing) return false
        var hasAudio = audioPath !== ""
        if (isProjectOnly) return hasAudio && labMidiPath !== ""
        var hasText = inputText.trim() !== ""
        return hasAudio && (hasText || textOptional) && _backendReady()
    }

    function _backendReady() {
        if (backend === "mfa") {
            if (!status.mfaInstalled) return false
            return status.modelAvailable(language)
        }
        return status.backendAvailable(backend)
    }

    // ══════════════════════════════════════════════════════════════
    //  LAYOUT
    // ══════════════════════════════════════════════════════════════
    Column {
        id: mainCol
        anchors { fill: parent; margins: 20 }
        spacing: 14

        // ── card header ───────────────────────────────────────────
        RowLayout {
            width: parent.width
            Label {
                id: cardTitle
                text: i18n.t("single_file_processing")
                font.pixelSize: 13; font.bold: true; color: "#eceff1"
            }
            Item { Layout.fillWidth: true }
            Button {
                id: ghBtn; flat: true; font.pixelSize: 11
                text: i18n.t("github_link")
                Material.foreground: "#80cbc4"
                onClicked: Qt.openUrlExternally(
                    "https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner-tsubaki")
            }
            Button {
                id: refreshBtn; flat: true; font.pixelSize: 11
                text: i18n.t("check_status")
                Material.foreground: "#80cbc4"
                onClicked: api.checkStatus()
            }
        }
        Rectangle { width: parent.width; height: 1; color: "#3a3f5c" }

        // ── 1. Audio file ─────────────────────────────────────────
        FormRow {
            id: audioRow; width: parent.width
            labelText: i18n.t("audio_file")
            FileDropZone {
                width: parent.width
                acceptExts: ["wav","mp3","flac","m4a","aac","ogg"]
                hint:    i18n.t("drop_audio_hint")
                tipText: i18n.t("audio_formats_tip")
                filePath: root.audioPath
                onFileSelected: (p) => { root.audioPath = p }
                onCleared:      () => { root.audioPath = "" }
            }
        }

        // ── 2. Aligner backend ────────────────────────────────────
        FormRow {
            id: backendRow; width: parent.width
            visible: !isProjectOnly
            labelText: i18n.t("aligner_backend")

            Column {
                width: parent.width; spacing: 4

                Repeater {
                    model: [
                        { val:"mfa",           label:"MFA",           descKey:"backend_mfa_desc"           },
                        { val:"whisperx",      label:"WhisperX",      descKey:"backend_whisperx_desc"      },
                        { val:"qwen3_asr",     label:"Qwen3-ASR",     descKey:"backend_qwen3_asr_desc"     },
                        { val:"qwen3_aligner", label:"Qwen3-FA",      descKey:"backend_qwen3_aligner_desc" },
                    ]
                    delegate: RowLayout {
                        required property var modelData
                        width: parent.width; spacing: 6
                        RadioButton {
                            text:    modelData.label
                            checked: root.backend === modelData.val
                            onClicked: {
                                root.backend = modelData.val
                                backendDesc.text = i18n.t(modelData.descKey)
                                textInput.placeholderText = i18n.t(
                                    (root.backend === "whisperx" || root.backend === "qwen3_asr")
                                    ? "text_placeholder_optional"
                                    : "text_placeholder_required"
                                )
                            }
                        }
						StatusBadge {
							property bool avail: modelData.val === "mfa"           ? status.mfaInstalled
												: modelData.val === "whisperx"      ? status.whisperxAvailable
												: modelData.val === "qwen3_asr"     ? status.qwen3AsrAvailable
												: modelData.val === "qwen3_aligner" ? status.qwen3AlignerAvailable
												: false
							ok:    avail
							label: avail ? "✓" : "✗"
							implicitWidth: 28
                        }
                    }
                }
                Label {
                    id: backendDesc
                    text: i18n.t("backend_mfa_desc")
                    color: "#607d8b"; font.pixelSize: 10; wrapMode: Text.WordWrap
                    width: parent.width
                }
            }
        }

        // ── 3. Aligner device ─────────────────────────────────────
        FormRow {
            id: deviceRow; width: parent.width
            visible: showDevice
            labelText: i18n.t("aligner_device")

            Column {
                width: parent.width; spacing: 4
                RowLayout {
                    spacing: 12
                    Repeater {
                        model: [
                            { val:"auto", key:"device_auto" },
                            { val:"cpu",  key:"device_cpu"  },
                            { val:"cuda", key:"device_cuda" },
                        ]
                        delegate: RadioButton {
                            required property var modelData
                            text:    i18n.t(modelData.key)
                            checked: root.alignerDevice === modelData.val
                            onClicked: {
                                root.alignerDevice = modelData.val
                                devDesc.text = i18n.t(
                                    modelData.val === "cuda"
                                        ? (root.backend === "whisperx" ? "whisperx_gpu_hint" : "qwen3_gpu_hint")
                                    : modelData.val === "cpu" ? "cpu_mode_hint"
                                    :                           "auto_device_hint"
                                )
                            }
                        }
                    }
                }
                Label {
                    id: devDesc; text: i18n.t("auto_device_hint")
                    color: "#607d8b"; font.pixelSize: 10; wrapMode: Text.WordWrap
                    width: parent.width
                }
            }
        }

        // ── 4. LAB / MIDI (project-only) ──────────────────────────
        FormRow {
            id: labRow; width: parent.width
            visible: isProjectOnly
            labelText: i18n.t("lab_midi_file")
            FileDropZone {
                width: parent.width
                acceptExts: ["lab","mid","midi"]
                hint:    i18n.t("drop_lab_hint")
                tipText: i18n.t("lab_midi_tip")
                filePath: root.labMidiPath
                onFileSelected: (p) => {
                    root.labMidiPath = p
                    var ext = p.split(".").pop().toLowerCase()
                    if (ext === "mid" || ext === "midi") {
                        var bpmVal = api.parseMidiBpm(p)
                        root.bpm        = bpmVal
                        root.midiLocked = true
                        midiBanner.visible = true
                    }
                }
                onCleared: () => {
                    root.labMidiPath = ""
                    root.midiLocked  = false
                    midiBanner.visible = false
                }
            }
        }

        // MIDI banner
        Rectangle {
            id: midiBanner
            width: parent.width; visible: false
            height: visible ? midiLbl.implicitHeight + 12 : 0
            color: "#1a2a1a"; radius: 4
            border.color: "#4caf50"; border.width: 1
            Label {
                id: midiLbl
                anchors { fill: parent; margins: 6 }
                text: i18n.t("midi_loaded_banner")
                color: "#81c784"; wrapMode: Text.WordWrap
            }
        }

        // ── 5. Text input ─────────────────────────────────────────
        FormRow {
            id: textRow; width: parent.width
            visible: !isProjectOnly
            labelText: i18n.t("input_text")

            Column {
                width: parent.width; spacing: 4
                ScrollView {
                    width: parent.width; height: 90
                    TextArea {
                        id: textInput
                        placeholderText: i18n.t("text_placeholder_required")
                        background: Rectangle {
                            color: "#141624"; radius: 4
                            border.color: textInput.activeFocus ? "#80cbc4" : "#3a3f5c"
                        }
                        color: "#eceff1"; font.pixelSize: 12
                        wrapMode: TextArea.Wrap
                        onTextChanged: root.inputText = text
                    }
                }
                Label {
                    text: i18n.tf("char_count", {"count": root.inputText.length})
                    color: "#607d8b"; font.pixelSize: 10
                }
            }
        }

        // ── 6. Language ───────────────────────────────────────────
        FormRow {
            id: langRow; width: parent.width
            visible: !isProjectOnly
            labelText: i18n.t("language_select")
            ComboBox {
                id: langCombo
                model: [
                    i18n.t("lang_cmn"), i18n.t("lang_eng"),
                    i18n.t("lang_jpn"), i18n.t("lang_kor"), i18n.t("lang_yue"),
                ]
                implicitWidth: 200
                property var codes: ["cmn","eng","jpn","kor","yue"]
                currentIndex: codes.indexOf(root.language)
                onActivated: root.language = codes[currentIndex]
            }
        }

        // ── 7. Processing mode ────────────────────────────────────
        FormRow {
            id: modeRow; width: parent.width
            labelText: i18n.t("processing_mode")
            Column {
                width: parent.width; spacing: 4
                RowLayout {
                    spacing: 12
                    Repeater {
                        model: [
                            { val:"mfa-only",     key:"mode_mfa_only",     desc:"mode_mfa_only_desc"     },
                            { val:"full",         key:"mode_full",         desc:"mode_full_desc"         },
                            { val:"project-only", key:"mode_project_only", desc:"mode_project_only_desc" },
                        ]
                        delegate: RadioButton {
                            required property var modelData
                            text:    i18n.t(modelData.key)
                            checked: root.mode === modelData.val
                            onClicked: {
                                root.mode = modelData.val
                                modeDesc.text = i18n.t(modelData.desc)
                            }
                        }
                    }
                }
                Label {
                    id: modeDesc
                    text: i18n.t("mode_mfa_only_desc")
                    color: "#607d8b"; font.pixelSize: 10; wrapMode: Text.WordWrap
                    width: parent.width
                }
            }
        }

        // ── 8. Output format ──────────────────────────────────────
        FormRow {
            id: fmtRow; width: parent.width
            visible: !isMfaOnly
            labelText: i18n.t("output_format")
            ComboBox {
                id: fmtCombo
                model: [i18n.t("format_sv"), i18n.t("format_utau")]
                property var vals: ["sv","utau"]
                implicitWidth: 280
                currentIndex: vals.indexOf(root.outputFormat)
                onActivated: root.outputFormat = vals[currentIndex]
            }
        }

        // ── 9. Phoneme conversion ─────────────────────────────────
        FormRow {
            id: phonemeRow; width: parent.width
            visible: isProjectOnly
            labelText: i18n.t("phoneme_conversion")
            Column {
                width: parent.width; spacing: 4
                RowLayout {
                    spacing: 8
                    Repeater {
                        model: [
                            { val:"none",     key:"phoneme_none",     desc:"phoneme_none_desc"     },
                            { val:"merge",    key:"phoneme_merge",    desc:"phoneme_merge_desc"    },
                            { val:"hiragana", key:"phoneme_hiragana", desc:"phoneme_hiragana_desc" },
                            { val:"katakana", key:"phoneme_katakana", desc:"phoneme_katakana_desc" },
                        ]
                        delegate: RadioButton {
                            required property var modelData
                            text:    i18n.t(modelData.key)
                            checked: root.phonemeMode === modelData.val
                            onClicked: {
                                root.phonemeMode = modelData.val
                                phonemeDesc.text = i18n.t(modelData.desc)
                            }
                        }
                    }
                }
                Label {
                    id: phonemeDesc
                    text: i18n.t("phoneme_none_desc")
                    color: "#607d8b"; font.pixelSize: 10; wrapMode: Text.WordWrap
                    width: parent.width
                }
            }
        }

        // ── 10. Track name ────────────────────────────────────────
        FormRow {
            id: titleRow; width: parent.width
            visible: !isMfaOnly
            labelText: i18n.t("track_name")
            TextField {
                id: titleField; text: "Project"; implicitWidth: 260
                color: "#eceff1"
                background: Rectangle {
                    color: "#1a1d2e"; radius: 4
                    border.color: titleField.activeFocus ? "#80cbc4" : "#3a3f5c"
                }
                onTextChanged: root.trackName = text
            }
        }

        // ── 11. Advanced settings ─────────────────────────────────
        CollapsibleSection {
            id: advSection
            visible: !isMfaOnly
            width: parent.width
            title: i18n.t("advanced_settings")

            // BPM
            RowLabel { labelText: i18n.t("bpm")
                SpinBox { from:20; to:300; value: root.bpm; enabled: !root.midiLocked
                    implicitWidth: 180; onValueModified: root.bpm = value } }

            // Base pitch
            RowLabel { labelText: i18n.t("base_pitch")
                RowLayout {
                    spacing: 8
                    SpinBox { id: pitchSpin; from:12; to:108; value: root.basePitch
                        implicitWidth: 180; enabled: !root.midiLocked
                        onValueModified: root.basePitch = value }
                    Label { text: api.midiToNoteName(root.basePitch)
                        color: "#80cbc4"; font.bold: true }
                }
            }

            // Pitch section header
            Label { text: i18n.t("pitch_control"); color: "#80cbc4"
                font.pixelSize: 11; font.bold: true }

            RowLabel { labelText: i18n.t("auto_note_pitch")
                CheckBox { id: autoPitchChk; text: i18n.t("auto_note_pitch_on")
                    checked: root.autoNotePitch; enabled: !root.midiLocked
                    onCheckedChanged: root.autoNotePitch = checked } }

            RowLabel { labelText: i18n.t("export_pitch_line")
                CheckBox { text: i18n.t("export_pitch_line_on")
                    checked: root.exportPitchLine
                    onCheckedChanged: root.exportPitchLine = checked } }

            // F0 method section header
            Label { text: i18n.t("f0_method_section"); color: "#80cbc4"
                font.pixelSize: 11; font.bold: true }

            RowLabel { labelText: i18n.t("f0_method")
                RowLayout {
                    spacing: 8
                    Repeater {
                        model: [
                            { val:"dio",     key:"f0_dio"     },
                            { val:"harvest", key:"f0_harvest" },
                            { val:"crepe",   key:"f0_crepe"   },
                            { val:"rmvpe",   key:"f0_rmvpe"   },
                        ]
                        delegate: RadioButton {
                            required property var modelData
                            text:    i18n.t(modelData.key)
                            checked: root.f0Method === modelData.val
                            onClicked: root.f0Method = modelData.val
                        }
                    }
                }
            }

            RowLabel { labelText: i18n.t("crepe_model_size")
                visible: root.f0Method === "crepe" || root.f0Method === "rmvpe"
                RowLayout {
                    spacing: 8
                    RadioButton { text: i18n.t("crepe_full"); checked: root.crepeModel === "full"
                        onClicked: root.crepeModel = "full" }
                    RadioButton { text: i18n.t("crepe_tiny"); checked: root.crepeModel === "tiny"
                        onClicked: root.crepeModel = "tiny" }
                }
            }

            RowLabel { labelText: i18n.t("f0_device")
                visible: root.f0Method === "crepe" || root.f0Method === "rmvpe"
                RowLayout {
                    spacing: 8
                    Repeater {
                        model: [
                            { val:"auto", key:"device_auto" },
                            { val:"cpu",  key:"device_cpu"  },
                            { val:"cuda", key:"device_cuda" },
                        ]
                        delegate: RadioButton {
                            required property var modelData
                            text:    i18n.t(modelData.key)
                            checked: root.f0Device === modelData.val
                            onClicked: root.f0Device = modelData.val
                        }
                    }
                }
            }

            RowLabel { labelText: i18n.t("precision")
                RowLayout {
                    spacing: 8
                    RadioButton { text: i18n.t("precision_single")
                        checked: root.precision === "single"; onClicked: root.precision = "single" }
                    RadioButton { text: i18n.t("precision_double")
                        checked: root.precision === "double"; onClicked: root.precision = "double" }
                }
            }

            RowLabel { labelText: i18n.t("f0_smooth")
                CheckBox { text: i18n.t("f0_smooth"); checked: root.f0Smooth
                    onCheckedChanged: root.f0Smooth = checked } }

            RowLabel { labelText: i18n.t("f0_smooth_window")
                RowLayout {
                    SpinBox { from:1; to:30; value: root.f0SmoothWindow; stepSize: 2
                        implicitWidth: 150; onValueModified: root.f0SmoothWindow = value }
                    Label { text: i18n.t("smooth_tip"); color: "#607d8b"; font.pixelSize: 10 }
                }
            }

            RowLabel { labelText: i18n.t("f0_floor_hz")
                SpinBox { from:40; to:200; value: root.f0Floor; textFromValue: (v) => v + " Hz"
                    implicitWidth: 180; onValueModified: root.f0Floor = value } }

            RowLabel { labelText: i18n.t("f0_ceil_hz")
                Column {
                    spacing: 2
                    SpinBox { from:300; to:2000; value: root.f0Ceil; textFromValue: (v) => v + " Hz"
                        implicitWidth: 180; onValueModified: root.f0Ceil = value }
                    Label { text: i18n.t("f0_range_tip"); color: "#607d8b"; font.pixelSize: 10 }
                }
            }
        }

        // ── 12. Action row ────────────────────────────────────────
        RowLayout {
            width: parent.width; spacing: 10

            Button {
                id: startBtn
                text: processing
                    ? i18n.tf("processing_btn", {"percent": root.progressValue})
                    : i18n.t("start_processing")
                enabled: root.canStart
                implicitHeight: 40
                font.pixelSize: 12; font.bold: true
                Material.background: "#00897b"
                Material.foreground: "#ffffff"
                onClicked: _doProcess()
            }
            Button {
                text: i18n.t("reset")
                implicitHeight: 40
                enabled: !processing
                onClicked: _doReset()
            }
            Label {
                text: i18n.t("system_not_ready_hint")
                color: "#f44336"; font.pixelSize: 10
                visible: !root.canStart && !processing && !isProjectOnly
                         && audioPath !== "" && !_backendReady()
            }
            Item { Layout.fillWidth: true }
        }

        // ── 13. Progress bar ──────────────────────────────────────
        ProgressBar {
            id: progressBar
            width: parent.width
            from: 0; to: 100; value: root.progressValue
            visible: processing
            Material.accent: "#00897b"
        }

        // ── 14. Error label ───────────────────────────────────────
        Rectangle {
            width: parent.width; visible: errorText !== ""
            height: visible ? errLbl.implicitHeight + 16 : 0
            color: "#4a1010"; radius: 4; border.color: "#b71c1c"; border.width: 1
            Label {
                id: errLbl
                anchors { fill: parent; margins: 8 }
                text: "❌  " + root.errorText
                color: "#ef9a9a"; wrapMode: Text.WordWrap
            }
        }

        // ── 15. Result panel ──────────────────────────────────────
        Column {
            id: resultPanel
            width: parent.width; spacing: 10
            visible: root.jobResult !== null

            Rectangle { width: parent.width; height: 1; color: "#3a3f5c" }

            Label {
                text: i18n.t("result_title")
                font.pixelSize: 12; font.bold: true; color: "#4caf50"
            }
            RowLayout {
                Label { id: resTime; color: "#b0bec5"; font.pixelSize: 11 }
                Label { id: resSegs; color: "#b0bec5"; font.pixelSize: 11 }
            }

            // Tabs
            TabBar {
                id: tabBar; width: parent.width
                TabButton { text: i18n.t("tab_lab_content") }
                TabButton { text: i18n.t("tab_file_info")   }
                TabButton { text: i18n.t("tab_details")     }
            }
            StackLayout {
                width: parent.width; currentIndex: tabBar.currentIndex

                // Tab 0 — LAB content
                Column {
                    spacing: 8
                    ScrollView {
                        width: parent.width; height: 200
                        TextArea {
                            id: labTextArea
                            readOnly: true; wrapMode: TextArea.Wrap
                            font.family: "Consolas,Courier New"; font.pixelSize: 10
                            color: "#a5d6a7"
                            background: Rectangle { color: "#141624"; radius: 4;
                                border.color: "#3a3f5c" }
                        }
                    }
                    RowLayout {
                        Button { text: i18n.t("copy_lab")
                            onClicked: {
                                labTextArea.selectAll()
                                labTextArea.copy()
                                labTextArea.deselect()
                            }
                        }
                        Button { text: i18n.t("download_lab"); onClicked: saveLabDialog.open() }
                    }
                }

                // Tab 1 — File info
                ScrollView {
                    width: parent.width; height: 200
                    TextArea {
                        id: infoTextArea
                        readOnly: true; wrapMode: TextArea.Wrap
                        font.pixelSize: 11; color: "#eceff1"
                        background: Rectangle { color: "#141624"; radius: 4;
                            border.color: "#3a3f5c" }
                    }
                }

                // Tab 2 — Stages detail
                Column {
                    spacing: 0
                    // header
                    RowLayout {
                        width: parent.width
                        Rectangle { width: 160; height: 30; color: "#1e2030"
                            Label { anchors.centerIn: parent; text: i18n.t("col_stage")
                                color: "#80cbc4"; font.bold: true } }
                        Rectangle { width: 100; height: 30; color: "#1e2030"
                            Label { anchors.centerIn: parent; text: i18n.t("col_status")
                                color: "#80cbc4"; font.bold: true } }
                        Rectangle { Layout.fillWidth: true; height: 30; color: "#1e2030"
                            Label { anchors { left: parent.left; leftMargin: 8; verticalCenter: parent.verticalCenter }
                                text: i18n.t("col_details"); color: "#80cbc4"; font.bold: true } }
                    }
                    Repeater {
                        model: stagesModel
                        delegate: RowLayout {
                            width: parent.width; spacing: 0
                            Rectangle { width: 160; height: 28; color: "#141624"
                                border.color: "#3a3f5c"
                                Label { anchors { left: parent.left; leftMargin: 8; verticalCenter: parent.verticalCenter }
                                    text: model.stage; color: "#eceff1"; font.pixelSize: 11 } }
                            Rectangle { width: 100; height: 28; color: model.statusColor
                                Label { anchors.centerIn: parent; text: model.statusText
                                    color: "#ffffff"; font.pixelSize: 11 } }
                            Rectangle { Layout.fillWidth: true; height: 28; color: "#141624"
                                border.color: "#3a3f5c"
                                Label { anchors { left: parent.left; leftMargin: 8; verticalCenter: parent.verticalCenter }
                                    text: model.detail; color: "#eceff1"; font.pixelSize: 11 } }
                        }
                    }
                }
            }

            // result action buttons
            RowLayout {
                Button {
                    id: dlProjBtn; text: i18n.t("download_project"); visible: false
                    Material.background: "#1565c0"; Material.foreground: "#ffffff"
                    onClicked: saveProjDialog.open()
                }
                Button { text: i18n.t("process_next"); onClicked: _doReset() }
            }
        }
    } // end mainCol

    // ══════════════════════════════════════════════════════════════
    //  STAGE LIST MODEL
    // ══════════════════════════════════════════════════════════════
    ListModel {
        id: stagesModel
        ListElement { stage:""; statusText:""; statusColor:"#607d8b"; detail:"" }
        ListElement { stage:""; statusText:""; statusColor:"#607d8b"; detail:"" }
        ListElement { stage:""; statusText:""; statusColor:"#607d8b"; detail:"" }
    }

    function _resetStages() {
        stagesModel.setProperty(0, "stage",  i18n.t("stage_alignment"))
        stagesModel.setProperty(1, "stage",  i18n.t("stage_f0"))
        stagesModel.setProperty(2, "stage",  i18n.t("stage_project"))
        for (var i = 0; i < 3; i++) {
            stagesModel.setProperty(i, "statusText",  i18n.t("status_waiting"))
            stagesModel.setProperty(i, "statusColor", "#607d8b")
            stagesModel.setProperty(i, "detail",      "—")
        }
    }

    function _setStage(row, statusKey, detail) {
        var colorMap = { "done":"#4caf50","running":"#ff9800","skipped":"#9e9e9e","failed":"#f44336","waiting":"#607d8b" }
        stagesModel.setProperty(row, "statusText",  i18n.t("status_" + statusKey))
        stagesModel.setProperty(row, "statusColor", colorMap[statusKey] || "#607d8b")
        stagesModel.setProperty(row, "detail",      detail)
    }

    // ══════════════════════════════════════════════════════════════
    //  DIALOGS
    // ══════════════════════════════════════════════════════════════
    FileDialog {
        id: saveLabDialog
        fileMode: FileDialog.SaveFile
        title: "Save LAB"
        nameFilters: ["LAB files (*.lab)"]
        currentFile: "alignment.lab"
        onAccepted: {
            var path = _urlToPath(selectedFile.toString())
            api.writeTextFile(path, labTextArea.text)
        }
    }

    FileDialog {
        id: saveProjDialog
        fileMode: FileDialog.SaveFile
        title: "Save Project"
        nameFilters: ["Project files (*.svp *.ustx)"]
        onAccepted: {
            var savePath = _urlToPath(selectedFile.toString())
            if (root.jobResult) {
                var pp = root.jobResult["project_path"] || root.jobResult["output_path"] || ""
                if (pp) {
                    dlProjBtn.enabled = false
                    api.downloadFile(pp.split("/").pop().split("\\").pop(), savePath, "proj")
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  API CONNECTIONS
    // ══════════════════════════════════════════════════════════════
    Connections {
        target: api

        function onJobProgress(pct, msg) {
            if (pct > 0) root.progressValue = pct
        }

        function onJobCompleted(result) {
            root.processing    = false
            root.progressValue = 100
            root.jobResult     = result
            root.errorText     = ""

            // populate LAB
            labTextArea.text = result["lab_content"] || ""

            // populate file info
            var t = i18n
            var lines = []
            if (result["lab_path"])
                lines.push(t.t("lab_path_label") + "  " + result["lab_path"])
            var pp = result["project_path"] || result["output_path"] || ""
            if (pp) {
                lines.push(t.t("project_path_label") + "  " + pp)
                var fmt = result["project_format"] || "sv"
                lines.push(t.t("output_format_label") + "  " + (fmt === "sv" ? t.t("sv_studio") : t.t("openutau")))
            }
            var cfg = result["config"] || {}
            if (Object.keys(cfg).length > 0) {
                lines.push("")
                lines.push(t.t("config_label"))
                lines.push(t.tf("cfg_bpm", {"v": cfg["bpm"] || 120}))
                var bp = cfg["base_pitch"] || 60
                lines.push(t.tf("cfg_base_pitch", {"note": api.midiToNoteName(bp), "midi": bp}))
                var on = t.t("state_on"); var off = t.t("state_off")
                lines.push(t.tf("cfg_auto_pitch",   {"state": cfg["auto_note_pitch"]    ? on : off}))
                lines.push(t.tf("cfg_export_pitch", {"state": cfg["export_pitch_line"]  ? on : off}))
                lines.push(t.tf("cfg_f0_method",    {"method": (cfg["f0_method"] || "?").toUpperCase()}))
                lines.push(t.tf("cfg_device",       {"device": cfg["f0_device"] || "auto"}))
                var prec = cfg["use_double_precision"] ? t.t("precision_double") : t.t("precision_single")
                lines.push(t.tf("cfg_precision",    {"prec": prec}))
            }
            infoTextArea.text = lines.join("\n")

            // summary labels
            var ms = result["processing_time"] || 0
            resTime.text = t.tf("processing_time", {"time": _fmtMs(ms)})
            var segs = result["segments"] || 0
            resSegs.text = segs > 0 ? t.tf("segments_label", {"count": segs}) : ""

            // stages
            _resetStages()
            if (root.mode === "mfa-only") {
                _setStage(0, "done",    segs + " segments")
                _setStage(1, "skipped", t.t("status_skipped"))
                _setStage(2, "skipped", t.t("status_skipped"))
            } else if (root.mode === "project-only") {
                _setStage(0, "skipped", t.t("status_skipped"))
                _setStage(1, "done",    "F0 done")
                _setStage(2, "done",    pp.split("/").pop() || "")
            } else {
                _setStage(0, "done", "alignment done")
                _setStage(1, "done", "F0 done")
                _setStage(2, "done", pp.split("/").pop() || "")
            }

            dlProjBtn.visible = pp !== ""
        }

        function onJobFailed(err) {
            root.processing = false
            root.errorText  = err
        }

        function onFileDownloaded(key, path) {
            if (key === "proj") dlProjBtn.enabled = true
        }
        function onFileDownloadFailed(key, err) {
            if (key === "proj") {
                dlProjBtn.enabled = true
                root.errorText    = err
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    //  ACTIONS
    // ══════════════════════════════════════════════════════════════
    function _doProcess() {
        if (!root.canStart) return

        var lab  = ""
        var midi = ""
        if (isProjectOnly) {
            var ext = root.labMidiPath.split(".").pop().toLowerCase()
            if (ext === "lab") lab  = root.labMidiPath
            else               midi = root.labMidiPath
        }

        root.processing    = true
        root.progressValue = 10
        root.jobResult     = null
        root.errorText     = ""
        _resetStages()

        var params = {
            "text":              root.inputText.trim(),
            "aligner_backend":   root.backend,
            "aligner_device":    root.alignerDevice,
            "output_format":     root.outputFormat,
            "project_title":     root.trackName || "Project",
            "bpm":               root.bpm,
            "base_pitch":        root.basePitch,
            "auto_note_pitch":   root.autoNotePitch,
            "export_pitch_line": root.exportPitchLine,
            "f0_method":         root.f0Method,
            "crepe_model":       root.crepeModel,
            "f0_device":         root.f0Device,
            "precision":         root.precision,
            "f0_smooth":         root.f0Smooth,
            "f0_smooth_window":  root.f0SmoothWindow,
            "f0_floor":          root.f0Floor,
            "f0_ceil":           root.f0Ceil,
            "phoneme_mode":      root.phonemeMode,
        }
        api.startJob(root.mode, root.audioPath, lab, midi, root.language, params)
    }

    function _doReset() {
        root.audioPath    = ""
        root.labMidiPath  = ""
        root.inputText    = ""
        root.trackName    = "Project"
        root.jobResult    = null
        root.errorText    = ""
        root.processing   = false
        root.progressValue = 0
        root.midiLocked   = false
        midiBanner.visible = false
        textInput.text    = ""
        titleField.text   = "Project"
        _resetStages()
    }

    // ══════════════════════════════════════════════════════════════
    //  HELPERS
    // ══════════════════════════════════════════════════════════════
    function _fmtMs(ms) {
        var s = Math.floor(ms / 1000)
        if (s < 60)   return s + "s"
        if (s < 3600) return Math.floor(s/60) + "m " + (s%60) + "s"
        return Math.floor(s/3600) + "h " + Math.floor((s%3600)/60) + "m " + (s%60) + "s"
    }
    function _urlToPath(url) {
        var s = url.toString()
        s = s.replace(/^file:\/\/\/([A-Za-z]:)/, "$1")
        s = s.replace(/^file:\/\//, "")
        return s
    }

    // ── init stages ───────────────────────────────────────────────
    Component.onCompleted: _resetStages()

    // ── i18n refresh ──────────────────────────────────────────────
    Connections {
        target: i18n
        function onLanguageChanged() {
            cardTitle.text  = i18n.t("single_file_processing")
            ghBtn.text      = i18n.t("github_link")
            refreshBtn.text = i18n.t("check_status")
            _resetStages()
        }
    }
}

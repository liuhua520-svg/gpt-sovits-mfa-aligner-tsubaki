// SystemStatusWidget.qml — bottom status panel
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Rectangle {
    id: root
    radius: 8; color: "#1e2030"
    border.color: "#3a3f5c"; border.width: 1
    implicitHeight: mainCol.implicitHeight + 28

    // per-language download state
    property var dlState: ({ cmn:false, eng:false, jpn:false, kor:false, yue:false })

    Connections {
        target: api
        function onModelDownloading(lang, isDownloading) {
            var d = Object.assign({}, root.dlState)
            d[lang] = isDownloading
            root.dlState = d
        }
    }

    Column {
        id: mainCol
        anchors { fill: parent; margins: 16 }
        spacing: 10

        // title
        Label {
            id: titleLbl
            text: i18n.t("system_status")
            font.pixelSize: 13; font.bold: true; color: "#80cbc4"
        }
        Rectangle { width: parent.width; height: 1; color: "#3a3f5c" }

        // ── MFA status row ────────────────────────────────────────
        GridLayout {
            width: parent.width
            columns: 2; columnSpacing: 24; rowSpacing: 10

            Label { id: mfaLbl; text: i18n.t("mfa_status"); color: "#b0bec5"; font.bold: true }
            StatusBadge {
                ok:    status.mfaInstalled
                label: status.mfaInstalled ? i18n.t("mfa_installed") : i18n.t("mfa_not_installed")
            }

            Label { id: verLbl; text: i18n.t("mfa_version"); color: "#b0bec5" }
            Label { text: status.mfaVersion; color: "#eceff1" }

            // ── language models ───────────────────────────────────
            Label {
                id: langModLbl; text: i18n.t("language_models")
                color: "#b0bec5"; font.bold: true
                Layout.alignment: Qt.AlignTop
            }
            Column {
                spacing: 4
                Repeater {
                    model: [
                        { code:"cmn", key:"model_cmn" },
                        { code:"eng", key:"model_eng" },
                        { code:"jpn", key:"model_jpn" },
                        { code:"kor", key:"model_kor" },
                        { code:"yue", key:"model_yue" },
                    ]
                    delegate: RowLayout {
                        required property var  modelData
                        required property int  index
                        spacing: 6

                        StatusBadge {
                            // 【修复】不要调用 status.modelAvailable() —— Slot 方法调用不会被 QML
                            // 绑定系统追踪依赖，updated 信号触发后这个表达式永远不会重新求值。
                            // 改为直接读取 status.models（带 notify=updated 的 Property），
                            // 下标访问会在每次绑定重新求值时拿到最新数据。
                            ok:    !!status.models[modelData.code]
                            label: i18n.t(modelData.key)
                        }
                        Button {
                            id: dlBtn
                            property bool dling: root.dlState[modelData.code] || false
                            text: dling ? i18n.t("downloading_btn") : i18n.t("download_btn")
                            enabled: !dling
                            visible: !status.models[modelData.code]
                            implicitWidth: 90; font.pixelSize: 11
                            Material.background: "#00695c"
                            Material.foreground: "#ffffff"
                            onClicked: api.downloadModel(modelData.code)
                        }
                    }
                }
            }

            // ── processing modules ────────────────────────────────
            Label {
                id: procLbl; text: i18n.t("processing_modules")
                color: "#b0bec5"; font.bold: true
                Layout.alignment: Qt.AlignTop
            }
            Column {
                spacing: 4
                StatusBadge { ok: status.pyworldAvailable; label: "PyWORLD (DIO/Harvest)" }
                StatusBadge { ok: status.crepeAvailable;   label: "CREPE"                 }
                StatusBadge { ok: status.rmvpeAvailable;   label: "RMVPE"                 }
            }

            // ── alt backends ──────────────────────────────────────
            Label {
                id: altLbl; text: i18n.t("alt_backends_section")
                color: "#b0bec5"; font.bold: true
                Layout.alignment: Qt.AlignTop
            }
            Column {
                spacing: 4
                StatusBadge { ok: status.whisperxAvailable;     label: "WhisperX"        }
                StatusBadge { ok: status.qwen3AsrAvailable;     label: "Qwen3-ASR-1.7B"  }
                StatusBadge { ok: status.qwen3AlignerAvailable; label: "Qwen3-FA-0.6B"   }
            }
        }

        // ── model cache dir ───────────────────────────────────────
        RowLayout {
            width: parent.width
            visible: status.cacheDir !== ""
            Label {
                id: cacheLbl; text: i18n.t("model_cache_dir")
                font.pixelSize: 11; color: "#b0bec5"
            }
            Label {
                text: status.cacheDir
                font.pixelSize: 11; color: "#80cbc4"
                wrapMode: Text.WrapAnywhere
                Layout.fillWidth: true
            }
        }
    }

    Connections {
        target: i18n
        function onLanguageChanged() {
            titleLbl.text  = i18n.t("system_status")
            mfaLbl.text    = i18n.t("mfa_status")
            verLbl.text    = i18n.t("mfa_version")
            langModLbl.text = i18n.t("language_models")
            procLbl.text   = i18n.t("processing_modules")
            altLbl.text    = i18n.t("alt_backends_section")
            cacheLbl.text  = i18n.t("model_cache_dir")
        }
    }
}

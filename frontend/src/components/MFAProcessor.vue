<template>
  <div class="processor-container">
    <el-card class="processor-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span class="card-title">📁 {{ t('processor.cardTitle') }}</span>
          <div class="header-actions">
            <el-tooltip :content="t('processor.githubTooltip')" placement="bottom">
              <el-button 
                link 
                @click="openGitHub"
                type="primary"
              >
                🔗 {{ t('processor.githubLink') }}
              </el-button>
            </el-tooltip>
            <el-tooltip :content="t('processor.checkStatus')" placement="bottom">
              <el-button link @click="refreshStatus" :loading="checkingStatus">
                🔄 {{ t('processor.checkStatus') }}
              </el-button>
            </el-tooltip>
          </div>
        </div>
      </template>

      <el-form :model="formData" label-position="top" class="processor-form">
        <el-form-item :label="t('processor.audioFile')">
          <el-upload
            :key="audioUploadKey"
            drag
            action="#"
            :auto-upload="false"
            :limit="1"
            :on-exceed="handleExceed"
            @change="handleAudioSelect"
            accept=".wav,.mp3,.flac,.m4a,.aac,.ogg"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              {{ t('processor.dragAudio') }}
            </div>
            <template #tip>
              <div class="el-upload__tip">
                {{ t('processor.supportedAudio') }}
              </div>
            </template>
          </el-upload>
          <div v-if="formData.audioFile" class="file-info">
            ✓ {{ formData.audioFile.name }} ({{ formatFileSize(formData.audioFile.size) }})
          </div>
        </el-form-item>

        <!-- 对齐后端选择器（project-only 模式不需要对齐） -->
        <el-form-item v-if="processingMode !== 'project-only'" :label="t('processor.backendLabel')">
          <el-radio-group v-model="alignerBackend">
            <el-radio value="mfa">
              <span>{{ t('processor.backendMfa') }}</span>
              <el-tag
                :type="systemStatus.mfa?.installed ? 'success' : 'danger'"
                size="small" style="margin-left:4px"
              >{{ systemStatus.mfa?.installed ? '✓' : '✗' }}</el-tag>
            </el-radio>
            <el-radio value="whisperx">
              <span>{{ t('processor.backendWhisperx') }}</span>
              <el-tag
                :type="alignerStatus['whisperx']?.available ? 'success' : 'info'"
                size="small" style="margin-left:4px"
              >{{ alignerStatus['whisperx']?.available ? '✓' : t('processor.backendStatusNeedInstall') }}</el-tag>
            </el-radio>
            <el-radio value="qwen3_asr">
              <span>{{ t('processor.backendQwen3Asr') }}</span>
              <el-tag
                :type="alignerStatus['qwen3_asr']?.available ? 'success' : 'info'"
                size="small" style="margin-left:4px"
              >{{ alignerStatus['qwen3_asr']?.available ? '✓' : t('processor.backendStatusNeedInstall') }}</el-tag>
            </el-radio>
            <el-radio value="qwen3_aligner">
              <span>{{ t('processor.backendQwen3Aligner') }}</span>
              <el-tag
                :type="alignerStatus['qwen3_aligner']?.available ? 'success' : 'info'"
                size="small" style="margin-left:4px"
              >{{ alignerStatus['qwen3_aligner']?.available ? '✓' : t('processor.backendStatusNeedInstall') }}</el-tag>
            </el-radio>
            <el-radio value="nemo_aligner">
              <span>{{ t('processor.backendNemoAligner') }}</span>
              <el-tag
                :type="alignerStatus['nemo_aligner']?.available ? 'success' : 'info'"
                size="small" style="margin-left:4px"
              >{{ alignerStatus['nemo_aligner']?.available ? '✓' : t('processor.backendStatusNeedInstall') }}</el-tag>
            </el-radio>
          </el-radio-group>
          <div class="help-text" style="margin-top:6px">
            <small v-if="alignerBackend === 'mfa'">
              🎯 {{ t('processor.backendMfaHelp') }}
            </small>
            <small v-else-if="alignerBackend === 'whisperx'">
              🤖 {{ t('processor.backendWhisperxHelp') }}
            </small>
            <small v-else-if="alignerBackend === 'qwen3_asr'">
              🌐 {{ t('processor.backendQwen3AsrHelp') }}
            </small>
            <small v-else-if="alignerBackend === 'qwen3_aligner'">
              📌 {{ t('processor.backendQwen3AlignerHelp') }}
            </small>
            <small v-else-if="alignerBackend === 'nemo_aligner'">
              🟩 {{ t('processor.backendNemoAlignerHelp') }}
            </small>
            <div v-if="alignerBackend !== 'mfa' && alignerStatus.models_dir"
                 style="margin-top:4px;color:#67c23a;font-size:12px">
              📁 {{ t('processor.modelCacheDir') }}：<code>{{ alignerStatus.models_dir }}</code>
            </div>
          </div>
          <el-alert
            v-if="alignerBackend !== 'mfa' && !alignerStatus[alignerBackend]?.available"
            type="warning" :closable="false" show-icon style="margin-top:8px"
          >
            <template #title>{{ alignerStatus[alignerBackend]?.message || t('processor.backendInstallHint') }}</template>
            <div style="font-size:12px;margin-top:4px">
              <span v-if="alignerBackend === 'whisperx'">{{ t('processor.packageHintWhisperx') }}</span>
              <span v-else-if="alignerBackend === 'nemo_aligner'">{{ t('processor.packageHintNemo') }}</span>
              <span v-else-if="alignerBackend === 'qwen3_aligner'">{{ t('processor.packageHintQwen3Aligner') }}</span>
              <span v-else-if="alignerBackend === 'qwen3_asr'">{{ t('processor.packageHintQwen3Asr') }}</span>
              <span v-else>{{ t('processor.packageHintTransformers') }} torchaudio accelerate</span>
            </div>
          </el-alert>
        </el-form-item>

        <!-- 对齐工具运行设备（仅非 MFA 后端显示） -->
        <el-form-item
          v-if="processingMode !== 'project-only' && alignerBackend !== 'mfa'"
          :label="t('processor.alignDevice')"
        >
          <el-radio-group v-model="advancedConfig.aligner_device">
            <el-radio label="auto">{{ t('processor.deviceAuto') }}</el-radio>
            <el-radio label="cpu">{{ t('processor.deviceCpu') }}</el-radio>
            <el-radio label="cuda">{{ t('processor.deviceCuda') }}</el-radio>
          </el-radio-group>
          <div class="help-text" style="margin-top:6px">
            <small v-if="advancedConfig.aligner_device === 'cuda' && alignerBackend === 'whisperx'">
              💡 {{ t('processor.deviceWhisperxGpu') }}
            </small>
            <small v-else-if="advancedConfig.aligner_device === 'cuda' && alignerBackend.startsWith('qwen3')">
              💡 {{ t('processor.deviceQwen3Gpu') }}
            </small>
            <small v-else-if="advancedConfig.aligner_device === 'cuda' && alignerBackend === 'nemo_aligner'">
              💡 {{ t('processor.deviceNemoGpu') }}
            </small>
            <small v-else-if="advancedConfig.aligner_device === 'cpu'">
              ⚠️ {{ t('processor.deviceCpuHelp') }}
            </small>
            <small v-else>
              {{ t('processor.deviceAutoHelp') }}
            </small>
          </div>
        </el-form-item>

        <!-- WhisperX 模型选择 -->
        <el-form-item
          v-if="processingMode !== 'project-only' && alignerBackend === 'whisperx'"
          :label="t('processor.whisperModel')"
        >
          <el-select v-model="advancedConfig.whisperx_model" style="width:240px">
            <el-option value="large-v3"       :label="t('processor.whisperModelLargeV3')" />
            <el-option value="large-v3-turbo" :label="t('processor.whisperModelLargeV3Turbo')" />
            <el-option value="large-v2"       :label="t('processor.whisperModelLargeV2')" />
            <el-option value="medium"         :label="t('processor.whisperModelMedium')" />
            <el-option value="small"          :label="t('processor.whisperModelSmall')" />
            <el-option value="base"           :label="t('processor.whisperModelBase')" />
            <el-option value="tiny"           :label="t('processor.whisperModelTiny')" />
          </el-select>
          <div class="help-text" style="margin-top:6px">
            <small v-if="advancedConfig.whisperx_model === 'large-v3'">
              🌟 {{ t('processor.whisperDescLargeV3') }}
            </small>
            <small v-else-if="advancedConfig.whisperx_model === 'large-v3-turbo'">
              ⚡ {{ t('processor.whisperDescLargeV3Turbo') }}
            </small>
            <small v-else-if="advancedConfig.whisperx_model === 'large-v2'">
              🔵 {{ t('processor.whisperDescLargeV2') }}
            </small>
            <small v-else>
              ⚠️ {{ t('processor.whisperDescSmall') }}
            </small>
          </div>
        </el-form-item>

		<!-- LAB / MIDI 单文件上传（仅 project-only 模式） -->
		<el-form-item v-if="processingMode === 'project-only'" :label="t('processor.labMidiFile')">
		  <el-upload
			:key="labMidiUploadKey"
			drag
			action="#"
			:auto-upload="false"
			:limit="1"
			:on-exceed="handleLabMidiExceed"
			@change="handleLabMidiChange"
			accept=".lab,.mid,.midi"
		  >
			<el-icon class="el-icon--upload"><UploadFilled /></el-icon>
			<div class="el-upload__text">
			  {{ t('processor.dragLabMidi') }}
			</div>
			<template #tip>
			  <div class="el-upload__tip">
				{{ t('processor.labMidiTip') }}
			  </div>
			</template>
		  </el-upload>

		  <div v-if="formData.labFile" class="file-info" style="margin-top:6px">
			📄 {{ formData.labFile.name }} ({{ formatFileSize(formData.labFile.size) }})
		  </div>
		  <div v-if="formData.midiFile" class="file-info midi-loaded" style="margin-top:4px">
			🎹 {{ formData.midiFile.name }} ({{ formatFileSize(formData.midiFile.size) }})
			<span v-if="midiInfo.loaded" class="midi-bpm-tag">BPM {{ midiInfo.bpm }}</span>
		  </div>

		  <el-alert
			v-if="midiLoaded"
			type="info"
			:closable="false"
			show-icon
			style="margin-top:8px"
		  >
			<template #title>{{ t('processor.midiImportedTitle') }}</template>
			<p style="margin:4px 0 0;font-size:12px;color:#606266">
			  🔒 {{ t('processor.midiImportedTip') }}
			</p>
		  </el-alert>
		</el-form-item>

		<el-form-item
		  v-if="processingMode !== 'project-only'"
		  :label="t('processor.inputText')"
		>
		  <el-input
			v-model="formData.text"
			type="textarea"
			:rows="4"
			style="width: 100%"
			:placeholder="
			  isTextOptional
				? t('processor.textPlaceholderOptional')
				: t('processor.textPlaceholderRequired')
			"
		  />
		  
		  <div class="help-text" style="margin-top: 6px; font-size: 12px; color: #909399; line-height: 1.4; width: 100%;">
			<span v-if="isTextOptional" style="color: #67c23a; font-weight: 500;">
			  ✓ {{ t('processor.textOptionalHint') }} | {{ t('processor.currentChars') }}{{ formData.text.length }}
			</span>

			<span v-else>
			  {{ t('processor.currentChars') }}{{ formData.text.length }}
			</span>
		  </div>
		</el-form-item>

        <el-form-item v-if="processingMode !== 'project-only'" :label="t('processor.language')">
          <el-select v-model="formData.language" :placeholder="t('processor.languagePlaceholder')">
            <el-option :label="t('processor.languageCmn')" value="cmn" />
            <el-option :label="t('processor.languageEng')" value="eng" />
            <el-option :label="t('processor.languageJpn')" value="jpn" />
            <el-option :label="t('processor.languageKor')" value="kor" />
            <el-option :label="t('processor.languageYue')" value="yue" />
          </el-select>
        </el-form-item>

        <!-- 英语单词级对齐：仅当语言非日语时显示 -->
        <el-form-item
          v-if="processingMode !== 'project-only' && formData.language !== 'jpn'"
          :label="t('processor.englishWordAlign')"
        >
          <el-switch v-model="englishWordAlign" />
          <span class="option-hint">
            {{ t('processor.englishWordAlignHint') }}
          </span>
        </el-form-item>

        <el-form-item :label="t('processor.processingMode')">
          <el-radio-group v-model="processingMode">
            <el-radio value="mfa-only">{{ t('processor.processingModeMfaOnly') }}</el-radio>
            <el-radio value="full">{{ t('processor.processingModeFull') }}</el-radio>
            <el-radio value="project-only">{{ t('processor.processingModeProjectOnly') }}</el-radio>
          </el-radio-group>
          <div class="mode-help">
            <small v-if="processingMode === 'mfa-only'">
              {{ t('processor.processingModeMfaOnlyHint', { backend: alignerBackendLabel }) }}
            </small>
            <small v-else-if="processingMode === 'full'">
              {{ t('processor.processingModeFullHint', { backend: alignerBackendLabel }) }}
            </small>
            <small v-else>
              {{ t('processor.processingModeProjectOnlyHint') }}
            </small>
          </div>
        </el-form-item>

        <el-form-item v-if="processingMode === 'full' || processingMode === 'project-only'" :label="t('processor.outputFormat')">
          <el-select v-model="formData.outputFormat" :placeholder="t('processor.outputFormat')">
            <el-option :label="t('processor.outputFormatSv')" value="sv" />
            <el-option :label="t('processor.outputFormatUtau')" value="utau" />
            <el-option :label="t('processor.outputFormatVsqx')" value="vsqx" />
          </el-select>
        </el-form-item>

        <el-form-item v-if="processingMode === 'project-only'" :label="t('processor.phonemeMode')">
          <el-radio-group v-model="formData.phonemeMode">
            <el-radio value="none">{{ t('processor.phonemeNone') }}</el-radio>
            <el-radio value="merge">{{ t('processor.phonemeMerge') }}</el-radio>
            <el-radio value="hiragana">{{ t('processor.phonemeHiragana') }}</el-radio>
            <el-radio value="katakana">{{ t('processor.phonemeKatakana') }}</el-radio>
          </el-radio-group>
          <div class="help-text">
            <small v-if="formData.phonemeMode === 'none'">
              {{ t('processor.phonemeNoneHint') }}
            </small>
            <small v-else-if="formData.phonemeMode === 'merge'">
              {{ t('processor.phonemeMergeHint') }}
            </small>
            <small v-else-if="formData.phonemeMode === 'hiragana'">
              {{ t('processor.phonemeHiraganaHint') }}
            </small>
            <small v-else>
              {{ t('processor.phonemeKatakanaHint') }}
            </small>
          </div>
          <div class="help-text" style="margin-top:4px">
            <small style="color:#909399">
              ⚠ {{ t('processor.phonemeWarning') }}
            </small>
          </div>
        </el-form-item>

        <el-form-item v-if="processingMode === 'full' || processingMode === 'project-only'" :label="t('processor.projectTitle')">
          <el-input
            v-model="formData.projectTitle"
            :placeholder="t('processor.projectTitlePlaceholder')"
            maxlength="248"
          />
        </el-form-item>

        <el-collapse v-if="processingMode === 'full' || processingMode === 'project-only'" accordion>
          <el-collapse-item :title="`⚙️ ${t('processor.advancedSettingsTitle')}`" name="advanced">
            <el-row :gutter="20">
              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.bpm')">
                  <el-input-number
                    v-model="advancedConfig.bpm"
                    :min="20"
                    :max="300"
                    :step="1"
                    controls-position="right"
                    :disabled="midiLoaded"
                  />
                  <span v-if="midiLoaded && midiInfo.loaded" class="midi-lock-tip">
                    🔒 {{ midiInfo.bpm }} ({{ t('processor.midiImportedTitle') }})
                  </span>
                  <span v-else-if="midiLoaded" class="midi-lock-tip">
                    🔒 {{ t('processor.midiImportedMore') }}
                  </span>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.basePitch')">
                  <div class="pitch-input-group">
                    <el-input-number
                      v-model="advancedConfig.base_pitch"
                      :min="12"
                      :max="108"
                      :step="1"
                      controls-position="right"
                      :disabled="midiLoaded"
                  />
                    <span class="pitch-name">{{ midiNoteToName(advancedConfig.base_pitch) }}</span>
                    <span v-if="midiLoaded" class="midi-lock-tip">🔒 {{ t('processor.midiImportedMore') }}</span>
                  </div>
                </el-form-item>
              </el-col>

              <el-col :xs="24">
                <el-divider>📈 {{ t('processor.pitchControl') }}</el-divider>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.autoNotePitch')">
                  <el-switch 
                    v-model="advancedConfig.auto_note_pitch"
                    :active-text="t('processor.autoNotePitchActive')"
                    :inactive-text="t('processor.autoNotePitchInactive')"
                    :disabled="midiLoaded"
                  />
                  <span v-if="midiLoaded" class="midi-lock-tip" style="display:block;margin-top:4px">
                    🔒 {{ t('processor.midiImportedTip') }}
                  </span>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.exportPitchLine')">
                  <el-switch 
                    v-model="advancedConfig.export_pitch_line"
                    :active-text="t('processor.exportPitchLineActive')"
                    :inactive-text="t('processor.exportPitchLineInactive')"
                  />
                </el-form-item>
              </el-col>

              <el-col :xs="24">
                <el-divider>{{ t('processor.f0RangeDivider') }}</el-divider>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.f0Method')">
                  <el-radio-group v-model="advancedConfig.f0_method" :disabled="!advancedConfig.export_pitch_line && !advancedConfig.auto_note_pitch">
                    <el-radio label="dio">
                      <span>{{ t('processor.f0Dio') }}</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                    <el-radio label="harvest">
                      <span>{{ t('processor.f0Harvest') }}</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                    <el-radio label="crepe" :disabled="(!advancedConfig.export_pitch_line && !advancedConfig.auto_note_pitch) || systemStatus.audio_processing?.f0_backends?.crepe?.available === false">
                      <span>{{ t('processor.f0Crepe') }}</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                    <el-radio label="rmvpe" :disabled="(!advancedConfig.export_pitch_line && !advancedConfig.auto_note_pitch) || systemStatus.audio_processing?.f0_backends?.rmvpe?.available === false">
                      <span>{{ t('processor.f0Rmvpe') }}</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                  </el-radio-group>
                  <p v-if="advancedConfig.f0_method === 'crepe' && systemStatus.audio_processing?.f0_backends?.crepe?.available === false" class="help-text">
                    ⚠ {{ t('processor.crepeDependencyMissing') }}
                  </p>
                  <p v-if="advancedConfig.f0_method === 'rmvpe' && systemStatus.audio_processing?.f0_backends?.rmvpe?.available === false" class="help-text">
                    ⚠ {{ t('processor.rmvpeModelMissing') }}
                  </p>
                </el-form-item>
              </el-col>

              <el-col v-if="advancedConfig.f0_method === 'crepe'" :xs="24" :sm="12">
                <el-form-item :label="t('processor.crepeModelSpec')">
                  <el-radio-group v-model="advancedConfig.crepe_model">
                    <el-radio label="full">{{ t('processor.crepeFull') }}</el-radio>
                    <el-radio label="tiny">{{ t('processor.crepeTiny') }}</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>

              <el-col v-if="advancedConfig.f0_method === 'crepe' || advancedConfig.f0_method === 'rmvpe'" :xs="24" :sm="12">
                <el-form-item :label="t('processor.f0Device')">
                  <el-radio-group v-model="advancedConfig.f0_device">
                    <el-radio label="auto">{{ t('processor.deviceAuto') }}</el-radio>
                    <el-radio label="cpu">{{ t('processor.deviceCpu') }}</el-radio>
                    <el-radio label="cuda">{{ t('processor.deviceCuda') }}</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.precision')">
                  <el-radio-group v-model="advancedConfig.precision">
                    <el-radio label="single">{{ t('processor.precisionSingle') }}</el-radio>
                    <el-radio label="double">{{ t('processor.precisionDouble') }}</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.f0Smooth')">
                  <el-switch 
                    v-model="advancedConfig.f0_smooth"
                    :active-text="t('processor.enabled')"
                    :inactive-text="t('processor.disabled')"
                    :disabled="!advancedConfig.export_pitch_line"
                  />
                </el-form-item>
              </el-col>

              <el-col v-if="advancedConfig.f0_smooth" :xs="24" :sm="12">
                <el-form-item :label="t('processor.smoothWindow')">
                  <el-input-number
                    v-model="advancedConfig.f0_smooth_window"
                    :min="1"
                    :max="21"
                    :step="2"
                    controls-position="right"
                    :disabled="!advancedConfig.export_pitch_line"
                  />
                  <span class="help-text">{{ t('processor.smoothWindowTip') }}</span>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.f0Floor')">
                  <el-input-number
                    v-model="advancedConfig.f0_floor"
                    :min="40"
                    :max="200"
                    :step="5"
                    controls-position="right"
                  />
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item :label="t('processor.f0Ceil')">
                  <el-input-number
                    v-model="advancedConfig.f0_ceil"
                    :min="300"
                    :max="1000"
                    :step="50"
                    controls-position="right"
                  />
                </el-form-item>
              </el-col>
			</el-row> <el-row :gutter="20">
				<el-col :span="24">
					<el-form-item
					  v-if="showWordPhonemeMap"
					  :label="t('processor.wordPhonemeMap')"
					>
					  <div class="word-phoneme-container">
						<el-switch v-model="wordPhonemeMap" />

					  </div>
					</el-form-item>
				  </el-col>
				</el-row>

				<el-row v-if="showWordPhonemeMap && wordPhonemeMap" :gutter="20">
				  <el-col :span="24">
					<el-form-item :label="t('processor.dictSource')">
					  <el-select v-model="dictSource" style="width: 240px">
						<el-option value="default" :label="t('processor.dictSourceDefault')" />
						<el-option value="synthesizerv" :label="t('processor.dictSourceSynthesizerV')" />
						<el-option value="vocaloid" :label="t('processor.dictSourceVocaloid')" />
					  </el-select>
					  <div class="dict-source-hint">{{ t('processor.dictSourceHint') }}</div>
					</el-form-item>
				  </el-col>
				</el-row>

            <el-alert type="info" :closable="false" show-icon class="settings-info">
              <template #title>💡 {{ t('processor.advancedHelpTitle') }}</template>
              <p><strong>BPM:</strong> {{ t('processor.advancedHelpBpm') }}</p>
              <p><strong>{{ t('processor.basePitch') }}:</strong> {{ t('processor.advancedHelpBasePitch') }}</p>
              <p><strong>{{ t('processor.autoNotePitch') }}:</strong> {{ t('processor.advancedHelpAutoNotePitch') }}</p>
              <p><strong>{{ t('processor.exportPitchLine') }}:</strong> {{ t('processor.advancedHelpExportPitchLine') }}</p>
              <p><strong>DIO / Harvest / CREPE / RMVPE：</strong>{{ t('processor.advancedHelpF0Method') }}</p>
              <p><strong>{{ t('processor.f0Device') }}：</strong>{{ t('processor.advancedHelpDevice') }}</p>
              <p><strong>{{ t('processor.f0Floor') }} / {{ t('processor.f0Ceil') }}:</strong> {{ t('processor.advancedHelpRange') }}</p>
            </el-alert>
          </el-collapse-item>
        </el-collapse>

        <el-form-item style="margin-top: 20px">
          <el-button
            type="primary"
            size="large"
            :loading="processing"
            @click="processAudio"
            :disabled="isSubmitDisabled"
          >
            <span v-if="!processing">🚀 {{ t('processor.startProcessing') }}</span>
            <span v-else>{{ t('processor.processing') }} {{ progressPercent }}%</span>
          </el-button>
          <el-button @click="reset" :disabled="processing">🔄 {{ t('processor.reset') }}</el-button>
          <span v-if="!isReady && processingMode !== 'project-only'" class="disabled-text">
            ({{ t('processor.systemReadyHint') }})
          </span>
        </el-form-item>

        <el-progress
          v-if="processing"
          :percentage="progressPercent"
          :indeterminate="true"
          class="progress-bar"
        />
      </el-form>

      <div v-if="result" class="result-section">
        <el-divider />

        <h3>✅ {{ t('processor.result') }}</h3>
        <div class="result-info">
          <el-row :gutter="20">
            <el-col :xs="24" :sm="12">
              <p><strong>{{ t('processor.processingTime') }}:</strong> {{ formatTime(result.processingTime) }}</p>
              <p v-if="result.labPath"><strong>{{ t('processor.labFile') }}:</strong> {{ getFileName(result.labPath) }}</p>
            </el-col>
            <el-col :xs="24" :sm="12">
              <p v-if="result.projectPath"><strong>{{ t('processor.projectFile') }}:</strong> {{ getFileName(result.projectPath) }}</p>
              <p v-if="result.segments"><strong>{{ t('processor.segmentCount') }}:</strong> {{ result.segments }}</p>
            </el-col>
          </el-row>
        </div>

        <el-tabs>
          <el-tab-pane v-if="result.labContent" :label="t('processor.labContentTab')">
            <el-input
              v-model="result.labContent"
              type="textarea"
              :rows="12"
              readonly
              class="output-text"
            />
            <div class="tab-actions">
              <el-button @click="copyLabToClipboard" size="small">
                📋 {{ t('processor.copyLab') }}
              </el-button>
              <el-button @click="downloadLab" size="small" type="success">
                📥 {{ t('processor.downloadLab') }}
              </el-button>
            </div>
          </el-tab-pane>

          <el-tab-pane v-if="result.projectPath" :label="t('processor.fileInfoTab')">
            <div class="file-info-box">
              <el-row :gutter="20">
                <el-col :xs="24" v-if="result.labPath">
                  <p><strong>{{ t('processor.labFile') }}:</strong></p>
                  <code>{{ result.labPath }}</code>
                </el-col>
                <el-col :xs="24">
                  <p><strong>{{ t('processor.projectFile') }}:</strong></p>
                  <code>{{ result.projectPath }}</code>
                </el-col>
                <el-col :xs="24">
                  <p><strong>{{ t('processor.outputFormat') }}:</strong>
                    {{
                      result.projectFormat === 'sv'   ? t('processor.outputFormatSv')   :
                      result.projectFormat === 'vsqx' ? t('processor.outputFormatVsqx') :
                                                        t('processor.outputFormatUtau')
                    }}
                  </p>
                  <p v-if="result.segments"><strong>{{ t('processor.segmentCount') }}:</strong> {{ result.segments }}</p>
                  <p v-if="result.config"><strong>{{ t('processor.processingConfig') }}:</strong></p>
                  <ul v-if="result.config">
                    <li>{{ t('processor.bpm') }}: {{ result.config.bpm }}</li>
                    <li>{{ t('processor.basePitch') }}: {{ midiNoteToName(result.config.base_pitch) }} (MIDI {{ result.config.base_pitch }})</li>
                    <li>{{ t('processor.autoNotePitch') }}: {{ result.config.auto_note_pitch ? t('processor.enabled') : t('processor.disabled') }}</li>
                    <li>{{ t('processor.exportPitchLine') }}: {{ result.config.export_pitch_line ? t('processor.enabled') : t('processor.disabled') }}</li>
                    <li>{{ t('processor.f0Method') }}: {{ result.config.f0_method?.toUpperCase?.() || result.config.f0_method }}</li>
                    <li v-if="result.config.f0_method === 'crepe'">{{ t('processor.crepeModelSpec') }}: {{ result.config.crepe_model }}</li>
                    <li v-if="result.config.f0_method === 'crepe' || result.config.f0_method === 'rmvpe'">{{ t('processor.f0Device') }}: {{ result.config.f0_device }}</li>
                    <li v-if="result.config.aligner_device !== undefined">{{ t('processor.alignDevice') }}: {{ result.config.aligner_device }}</li>
                    <li v-if="result.whisperxModel">{{ t('processor.whisperModel') }}: {{ result.whisperxModel }}</li>
                    <li>{{ t('processor.precision') }}: {{ result.config.use_double_precision ? t('processor.precisionDouble') : t('processor.precisionSingle') }}</li>
                  </ul>
                </el-col>
              </el-row>
            </div>
          </el-tab-pane>

          <el-tab-pane :label="t('processor.stageTab')">
            <div class="details-box">
              <el-table :data="processingDetails" stripe style="width: 100%">
                <el-table-column prop="stage" :label="t('processor.processingStage')" width="200" />
                <el-table-column prop="status" :label="t('processor.status')" width="100">
                  <template #default="{ row }">
                    <el-tag v-if="row.status === '完成'" type="success">{{ t('processor.statusDone') }}</el-tag>
                    <el-tag v-else-if="row.status === '跳过'" type="warning">{{ t('processor.statusSkipped') }}</el-tag>
                    <el-tag v-else-if="row.status === '等待'" type="info">{{ t('processor.statusWaiting') }}</el-tag>
                    <el-tag v-else-if="row.status === '进行中'" type="warning">{{ t('processor.statusProcessing') }}</el-tag>
                    <el-tag v-else type="info">{{ row.status }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="message" :label="t('processor.detail')" show-overflow-tooltip />
              </el-table>
            </div>
          </el-tab-pane>
        </el-tabs>

        <div class="action-buttons">
          <el-button v-if="result.labContent" type="success" @click="downloadLab" size="large">
            📥 {{ t('processor.downloadLabFile') }}
          </el-button>
          <el-button 
            v-if="result.projectPath" 
            type="success" 
            @click="downloadProject" 
            size="large"
            :loading="downloadingProject"
          >
            📥 {{ t('processor.downloadProjectFile') }}
          </el-button>
          <el-button v-if="result.labContent" @click="copyLabToClipboard" size="large">
            📋 {{ t('processor.copyLabContent') }}
          </el-button>
          <el-button type="info" @click="newProcess" size="large">
            🔄 {{ t('processor.processNext') }}
          </el-button>
        </div>
      </div>

      <div v-if="error" class="error-section">
        <el-alert
          :title="`${t('processor.errorPrefix')}: ${error}`"
          type="error"
          :closable="true"
          @close="error = ''"
          show-icon
        />
      </div>
    </el-card>

    <div v-if="systemStatus" class="status-box">
      <el-card shadow="hover">
        <template #header>
          <span>🔧 {{ t('app.systemStatus') }}</span>
        </template>

        <el-row :gutter="20">
          <el-col :xs="24" :sm="12">
            <div class="status-item">
              <span class="label">{{ t('processor.mfaStatus') }}:</span>
              <el-tag :type="systemStatus.mfa?.installed ? 'success' : 'danger'" size="large">
                {{ systemStatus.mfa?.installed ? `✓ ${t('processor.available')}` : `✗ ${t('processor.notInstalled')}` }}
              </el-tag>
            </div>
            <div v-if="systemStatus.mfa?.installed" class="status-item">
              <span class="label">{{ t('processor.version') }}:</span>
              <span>{{ systemStatus.mfa?.version }}</span>
            </div>
          </el-col>

          <el-col :xs="24" :sm="12">
            <div class="label">{{ t('processor.modelStatus') }}:</div>
            <div class="model-list">
              <div v-for="(downloaded, lang) in normalizedModels" :key="lang" class="model-item">
                <el-tag :type="downloaded ? 'success' : 'warning'" size="small">
                  {{ lang.toUpperCase() }}: {{ downloaded ? '✓' : '✗' }}
                </el-tag>
                <el-button
                  v-if="!downloaded"
                  link
                  size="small"
                  @click="downloadModel(lang)"
                  :loading="downloadingLangs.includes(lang)"
                >
                  {{ t('processor.download') }}
                </el-button>
              </div>
            </div>
          </el-col>

          <el-col :xs="24">
            <div class="label">{{ t('processor.processingModules') }}:</div>
            <div class="model-list">
              <div class="model-item">
                <el-tag 
                  :type="systemStatus.audio_processing?.pyworld_available ? 'success' : 'warning'" 
                  size="small"
                >
                  PyWORLD (DIO/Harvest): {{ systemStatus.audio_processing?.pyworld_available ? '✓' : '✗' }}
                </el-tag>
              </div>
              <div class="model-item">
                <el-tag
                  :type="systemStatus.audio_processing?.f0_backends?.crepe?.available ? 'success' : 'info'"
                  size="small"
                >
                  CREPE: {{ systemStatus.audio_processing?.f0_backends?.crepe?.available ? '✓' : '✗' }}
                </el-tag>
              </div>
              <div class="model-item">
                <el-tag
                  :type="systemStatus.audio_processing?.f0_backends?.rmvpe?.available ? 'success' : 'info'"
                  size="small"
                >
                  RMVPE: {{ systemStatus.audio_processing?.f0_backends?.rmvpe?.available ? '✓' : '✗' }}
                </el-tag>
              </div>
            </div>
          </el-col>

          <el-col :xs="24">
            <div class="label">{{ t('processor.altBackends') }}:</div>
            <div class="model-list">
              <div v-for="(info, key) in altBackends" :key="key" class="model-item">
                <el-tag :type="info.available ? 'success' : 'info'" size="small">
                  {{ key === 'whisperx' ? t('processor.backendWhisperx')
                     : key === 'qwen3_asr' ? 'Qwen3-ASR-1.7B'
                     : key === 'qwen3_aligner' ? 'Qwen3-FA-0.6B'
                     : key === 'nemo_aligner' ? t('processor.backendNemoAligner')
                     : key }}:
                  {{ info.available ? `✓ ${t('processor.available')}` : `✗ ${t('processor.notInstalled')}` }}
                </el-tag>
                <span v-if="!info.available" class="help-text" style="font-size:11px;margin-left:6px">
                  {{ key === 'whisperx' ? t('processor.packageHintWhisperx')
                     : key === 'nemo_aligner' ? t('processor.packageHintNemo')
                     : key === 'qwen3_aligner' ? t('processor.packageHintQwen3Aligner')
                     : key === 'qwen3_asr' ? t('processor.packageHintQwen3Asr')
                     : t('processor.packageHintTransformers') }}
                </span>
              </div>
            </div>
            <div v-if="alignerStatus.models_dir" style="margin-top:6px;font-size:12px;color:#909399">
              📁 {{ t('processor.modelsLocation') }}：<code style="color:#67c23a">{{ alignerStatus.models_dir }}</code>
              <span style="margin-left:6px">{{ t('processor.modelsLocationHint') }}</span>
            </div>
          </el-col>
        </el-row>
      </el-card>
    </div>

    <div v-if="systemStatus && !systemStatus.mfa?.installed" class="warning-box">
      <el-alert type="error" :closable="false" show-icon>
        <template #title>❌ {{ t('processor.mfaNotInstalledTitle') }}</template>
        <p>{{ t('processor.mfaNotInstalledBody') }}</p>
        <code>pip install montreal-forced-aligner</code>
        <p style="margin-top: 10px">{{ t('processor.mfaNotInstalledMore') }}</p>
      </el-alert>
    </div>

    <div v-if="systemStatus && systemStatus.mfa?.installed && !isReady && processingMode !== 'project-only'" class="warning-box">
      <el-alert type="warning" :closable="false" show-icon>
        <template #title>⚠️ {{ t('processor.notReadyTitle') }}</template>
        <p>{{ t('processor.notReadyBody') }}</p>
      </el-alert>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, InfoFilled } from '@element-plus/icons-vue'
import { useI18n } from 'vue-i18n'

const emit = defineEmits<{
  (e: 'status-changed', status: SystemStatus): void
}>()

const { t, locale } = useI18n()

type ProcessingMode = 'mfa-only' | 'full' | 'project-only'

interface FormData {
  audioFile: File | null
  labFile: File | null
  midiFile: File | null       // MIDI 文件（仅 project-only 模式）
  text: string
  language: string
  outputFormat: string
  projectTitle: string
  phonemeMode: 'none' | 'merge' | 'hiragana' | 'katakana'
}

interface AdvancedConfig {
  bpm: number
  base_pitch: number
  auto_note_pitch: boolean
  export_pitch_line: boolean
  f0_method: 'dio' | 'harvest' | 'crepe' | 'rmvpe'
  f0_device: 'auto' | 'cpu' | 'cuda'
  aligner_device: 'auto' | 'cpu' | 'cuda'  // WhisperX / Qwen3 对齐工具运行设备
  whisperx_model: string                    // WhisperX Whisper 模型选择
  crepe_model: 'full' | 'tiny'
  precision: 'single' | 'double'
  f0_smooth: boolean
  f0_smooth_window: number
  f0_floor: number
  f0_ceil: number
}

interface F0BackendStatus {
  available: boolean
  torch?: boolean
  torchcrepe?: boolean
  cuda?: boolean
  model_path?: string
  model_found?: boolean
}

interface SystemStatus {
  mfa?: {
    installed: boolean
    version: string
    models?: Record<string, boolean>
  }
  audio_processing?: {
    pyworld_available: boolean
    supported_formats: string[]
    f0_backends?: {
      dio?: F0BackendStatus
      harvest?: F0BackendStatus
      crepe?: F0BackendStatus
      rmvpe?: F0BackendStatus
    }
  }
  alt_aligners?: Record<string, { available: boolean; message: string; requires_text?: boolean }>
}

// 核心表单与模式状态（合并唯一声明）
const processingMode = ref<ProcessingMode>('mfa-only')
const englishWordAlign = ref<boolean>(false)  // 英语单词级对齐（不做 ARPABET 音素拆分）
const wordPhonemeMap   = ref<boolean>(false)  // 英语单词 → 音素映射（SVP phonemes / VSQX <p lock="1">）
const dictSource        = ref<string>('default')  // 单词→音素词典来源："default"/"synthesizerv"/"vocaloid"
const alignerBackend = ref<string>('mfa')   // 对齐后端选择
const alignerStatus = ref<Record<string, any>>({
  whisperx:      { available: false, message: t('processor.backendStatusChecking') },
  qwen3_asr:     { available: false, message: t('processor.backendStatusChecking') },
  qwen3_aligner: { available: false, message: t('processor.backendStatusChecking') },
  nemo_aligner:  { available: false, message: t('processor.backendStatusChecking') },
})

const formData = ref<FormData>({
  audioFile: null,
  labFile: null,
  midiFile: null,
  text: '',
  language: 'cmn',
  outputFormat: 'sv',
  projectTitle: 'Project',
  phonemeMode: 'none'
})

const advancedConfig = ref<AdvancedConfig>({
  bpm: 120,
  base_pitch: 60,
  auto_note_pitch: true,
  export_pitch_line: true,
  f0_method: 'dio',
  f0_device: 'auto',
  aligner_device: 'auto',
  whisperx_model: 'large-v3',
  crepe_model: 'full',
  precision: 'double',
  f0_smooth: true,
  f0_smooth_window: 5,
  f0_floor: 71,
  f0_ceil: 800
})

const processing = ref(false)
const progressPercent = ref(0)
const result = ref<any>(null)
const error = ref('')
const checkingStatus = ref(false)
const downloadingLangs = ref<string[]>([])
const downloadingProject = ref(false)

const systemStatus = ref<SystemStatus>({
  mfa: {
    installed: false,
    version: 'unknown',
    models: { cmn: false, eng: false, jpn: false, kor: false, yue: false }
  },
  audio_processing: {
    pyworld_available: false,
    supported_formats: [],
    f0_backends: {
      dio: { available: true },
      harvest: { available: true },
      crepe: { available: false },
      rmvpe: { available: false }
    }
  }
})

const processingDetails = ref<any[]>([
  { stage: t('processor.stageAlign'), status: '等待', message: t('processor.stagePrepareAlign') },
  { stage: t('processor.stageF0'), status: '等待', message: t('processor.stageExtractF0') },
  { stage: t('processor.stageProject'), status: '等待', message: t('processor.stageGenerateProject') },
])

const currentJobId = ref<string>('')
let jobPollTimer: number | null = null

// MIDI 导入状态
const midiInfo = ref<{ bpm: number; loaded: boolean }>({ bpm: 120, loaded: false })
const labMidiUploadKey = ref(0)
const audioUploadKey = ref(0)

const midiLoaded = computed(() => processingMode.value === 'project-only' && !!formData.value.midiFile)
const selectedNotationFile = computed(() => formData.value.labFile || formData.value.midiFile)

// VSQX 歌手名 / ID：按处理模式 + 语种自动切换
// project-only 无语种选择，固定使用日语歌手声库
const vsqxSingerConfig = computed((): { name: string; id: string } => {
  if (processingMode.value === 'project-only') {
    return { name: 'MIKU_V4X_Original_EVEC', id: 'BCNFCY43LB2LZCD4' }
  }
  // full 模式：按语种映射
  switch (formData.value.language) {
    case 'eng': return { name: 'MIKU_V4_English',         id: 'BMLTD846MLYP2MEK' }
    case 'jpn': return { name: 'MIKU_V4X_Original_EVEC', id: 'BCNFCY43LB2LZCD4' }
    case 'kor': return { name: 'SeeU_SV01_KOR',           id: 'BX77CNBZLBPHZX97' }
    default:    return { name: 'MIKU_V4_Chinese',         id: 'BNGE7CP7EMTRSNC3' }  // cmn / yue
  }
})

// 计算属性
const normalizedModels = computed(() => {
  const defaultModels = { cmn: false, eng: false, jpn: false, kor: false, yue: false }
  if (!systemStatus.value.mfa?.models || typeof systemStatus.value.mfa.models !== 'object') {
    return defaultModels
  }
  return { ...defaultModels, ...systemStatus.value.mfa.models }
})

const isReady = computed(() => {
  // 替代后端不依赖 MFA 模型，只要后端可用或是 MFA 时检查模型
  if (alignerBackend.value !== 'mfa') {
    return alignerStatus.value[alignerBackend.value]?.available ?? false
  }
  return !!(systemStatus.value.mfa?.installed && normalizedModels.value[formData.value.language as keyof typeof normalizedModels.value])
})

// WhisperX / Qwen3-ASR 支持纯 ASR 模式（文本可选）
const isTextOptional = computed(() =>
  ['whisperx', 'qwen3_asr'].includes(alignerBackend.value)
)

// 控制整个表单项是否显示
// 【历史 bug 修复】原逻辑写死要求 processingMode === 'full'，导致
// project-only 模式下选择 vsqx/sv 输出格式时，该开关永远不出现——
// project-only 模式下并不存在"对齐"环节，本就不该被 englishWordAlign
// 约束；只有 full 模式（触发 MFA/WhisperX 等对齐）下才需要这个前置条件，
// 因为 word_phoneme_map 假设 LAB 的每个 label 是完整单词而非拆分后的
// 单个音素（这正是 englishWordAlign 控制的行为）。
const showWordPhonemeMap = computed(() => {
  const format = formData.value.outputFormat?.toLowerCase() || ''
  const isSupportedFormat = format.includes('sv') || format.includes('vsqx')
  const isProjectProducingMode =
    processingMode.value === 'full' || processingMode.value === 'project-only'
  const alignGuardOk =
    processingMode.value === 'full' ? englishWordAlign.value : true

  return isProjectProducingMode && isSupportedFormat && alignGuardOk
})

// 根据格式动态返回提示文本
const wordPhonemeHint = computed(() => {
  const format = formData.value.outputFormat?.toLowerCase() || ''
  
  if (format.includes('vsqx')) {
    return t('processor.wordPhonemeMapHintVsqx')
  }
  
  return t('processor.wordPhonemeMapHintSvp')
})

// alignerStatus 去掉 models_dir 字段，只保留后端对象供 v-for 使用
const altBackends = computed(() => {
  const { models_dir: _md, ...backends } = alignerStatus.value as any
  return backends as Record<string, any>
})

const alignerBackendLabel = computed(() => {
  void locale.value
  const labels: Record<string, string> = {
    mfa: t('processor.backendMfa'),
    whisperx: t('processor.backendWhisperx'),
    qwen3_asr: t('processor.backendQwen3Asr'),
    qwen3_aligner: t('processor.backendQwen3Aligner'),
    nemo_aligner: t('processor.backendNemoAligner'),
  }
  return labels[alignerBackend.value] || alignerBackend.value
})

watch(alignerBackend, (backend) => {
  if (processingMode.value === 'project-only' && ['whisperx', 'qwen3_asr', 'qwen3_aligner', 'nemo_aligner'].includes(backend)) {
    processingMode.value = 'mfa-only'
  }
})

// 根据不同模式控制提交按钮的禁用状态
const isSubmitDisabled = computed(() => {
  if (processingMode.value === 'project-only') {
    return !formData.value.audioFile || (!formData.value.labFile && !formData.value.midiFile)
  }
  const noText = !formData.value.text.trim() && !isTextOptional.value
  return !formData.value.audioFile || noText || !isReady.value
})

const handleLabMidiExceed = () => {
  ElMessage.error(t('processor.chooseOneFile'))
}

const handleLabMidiChange = (file: any) => {
  const raw: File | null = file?.raw || null
  if (!raw) return

  const ext = raw.name.toLowerCase().split('.').pop() || ''

  // 先清空，确保只保留一个文件
  formData.value.labFile = null
  formData.value.midiFile = null
  midiInfo.value = { bpm: 120, loaded: false }

  if (ext === 'lab') {
    formData.value.labFile = raw
    labMidiUploadKey.value += 1
    return
  }

  if (ext === 'mid' || ext === 'midi') {
    formData.value.midiFile = raw
    extractMidiBpm(raw).then(({ bpm }) => {
      midiInfo.value = { bpm, loaded: true }
    })
    labMidiUploadKey.value += 1
    return
  }

  ElMessage.error(t('processor.onlySupportNotation'))
  labMidiUploadKey.value += 1
}

onMounted(() => {
  checkSystemStatus()
})

// 辅助工具函数
const midiNoteToName = (note: number): string => {
  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  const octave = Math.floor(note / 12) - 1
  return `${notes[note % 12]}${octave}`
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
}

const formatTime = (ms: number): string => {
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`
  return `${seconds}s`
}

const getFileName = (path: string): string => {
  return path.split(/[\\/]/).pop() || path
}

const clearJobPolling = () => {
  if (jobPollTimer !== null) {
    window.clearTimeout(jobPollTimer)
    jobPollTimer = null
  }
}

const resetProcessingSteps = () => {
  const stageLabel = alignerBackendLabel.value
  processingDetails.value = [
    { stage: `1. ${stageLabel}`, status: '等待', message: t('processor.stagePrepareAlign') },
    { stage: t('processor.stageF0'), status: '等待', message: t('processor.stageExtractF0') },
    { stage: t('processor.stageProject'), status: '等待', message: t('processor.stageGenerateProject') }
  ]
}

const updateProcessingStep = (index: number, status: string, message: string) => {
  if (!processingDetails.value[index]) return
  processingDetails.value[index] = { ...processingDetails.value[index], status, message }
}

const extractProjectPath = (payload: any): string => {
  return payload?.project_file || payload?.projectPath || payload?.project_path || payload?.output_path || payload?.svp_path || payload?.ustx_path || ''
}

const SIL_PHONES = new Set(['sp', 'spn', 'sil', 'silence', 'pau', 'breath', 'noise', 'ap', 'blank'])

// 统计 LAB 文件中的非静音标注段数
const countLabSegments = (labContent: string): number => {
  if (!labContent) return 0
  return labContent.trim().split('\n').filter(line => {
    const parts = line.trim().split(/\s+/)
    const phone = (parts[2] || '').toLowerCase()
    return phone && !SIL_PHONES.has(phone)
  }).length
}

const normalizeResult = (payload: any) => {
  const projectPath = extractProjectPath(payload)
  return {
    labContent: payload?.lab_content || payload?.labContent || '',
    processingTime: payload?.processing_time || payload?.processingTime || 0,
    labPath: payload?.lab_path || payload?.labPath || '',
    projectPath,
    projectFormat: payload?.project_format || payload?.projectFormat || formData.value.outputFormat,
    segments: payload?.segments || 0,
    whisperxModel: alignerBackend.value === 'whisperx' ? advancedConfig.value.whisperx_model : undefined,
    config: payload?.config || {
      bpm: advancedConfig.value.bpm,
      base_pitch: advancedConfig.value.base_pitch,
      auto_note_pitch: advancedConfig.value.auto_note_pitch,
      export_pitch_line: advancedConfig.value.export_pitch_line,
      f0_method: advancedConfig.value.f0_method,
      f0_device: advancedConfig.value.f0_device,
      aligner_device: advancedConfig.value.aligner_device,
      crepe_model: advancedConfig.value.crepe_model,
      use_double_precision: advancedConfig.value.precision === 'double'
    }
  }
}

// 异步任务轮询（通用版，不含模式相关的步骤更新）
const waitForJobFinished = (jobId: string): Promise<any> => {
  clearJobPolling()
  currentJobId.value = jobId

  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(`/api/pipeline/job/${jobId}`)
        const data = await res.json()

        if (!res.ok || !data.success) {
          throw new Error(data.error || t('processor.jobStatusFailed'))
        }

        const job = data.job || {}

        if (job.status === 'done') {
          resolve(job.result || job)
          return
        } else if (job.status === 'failed') {
          throw new Error(job.error || t('processor.jobFailed'))
        }
        // queued / running: 继续轮询

        jobPollTimer = window.setTimeout(tick, 1500)
      } catch (e) {
        reject(e)
      }
    }
    tick()
  })
}

// 后端 API 交互
const checkSystemStatus = async () => {
  checkingStatus.value = true
  try {
    const [pipelineRes, alignerRes] = await Promise.all([
      fetch('/api/pipeline/status'),
      fetch('/api/aligner/status'),
    ])
    const pipelineData = await pipelineRes.json()
    if (pipelineData.success) {
      systemStatus.value = pipelineData.status
      // 同步 alt_aligners 到 alignerStatus（如果 pipeline/status 已经包含了）
      if (pipelineData.status?.alt_aligners) {
        alignerStatus.value = pipelineData.status.alt_aligners
      }
    }
    const alignerData = await alignerRes.json()
    if (alignerData.success && alignerData.backends) {
      const { mfa: _mfa, ...altBacks } = alignerData.backends
      alignerStatus.value = altBacks
    }
    // 【修复】把已经拿到的 systemStatus 直接通过事件传给父组件，
    // 而不是只发一个空事件让父组件自己再 fetch 一次 /api/pipeline/status。
    // 否则父组件（右上角"系统就绪"标签）和这里（底部"系统状态"面板）
    // 永远是两次独立的网络请求结果，天然就会有先后顺序差 + 偶尔不一致。
    emit('status-changed', systemStatus.value)
  } catch (e) {
    console.warn('无法检查系统状态:', e)
    ElMessage.warning(t('processor.backendConnectionFailed'))
  } finally {
    checkingStatus.value = false
  }
}

const refreshStatus = async () => {
  await checkSystemStatus()
  ElMessage.success(t('processor.backendRefreshSuccess'))
}

const openGitHub = () => {
  window.open('https://github.com/liuhua520-svg/SVS-Lab-Aligner', '_blank')
}

const handleExceed = () => {
  ElMessage.error(t('processor.chooseOneUpload'))
}

const handleAudioSelect = (file: any) => {
  const raw: File | null = file?.raw || null
  if (!raw) return

  formData.value.audioFile = raw
  error.value = ''
}

/**
 * 从 MIDI 文件的二进制内容中解析第一个 set_tempo 事件，换算成 BPM。
 * 纯浏览器端解析，不需要额外库。
 * MIDI Tempo meta event 格式: 0xFF 0x51 0x03 <3-byte microseconds>
 */
const extractMidiBpm = (file: File): Promise<{ bpm: number }> => {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const buf = new Uint8Array(e.target!.result as ArrayBuffer)
        let bpm = 120.0
        for (let i = 0; i < buf.length - 5; i++) {
          if (buf[i] === 0xFF && buf[i + 1] === 0x51 && buf[i + 2] === 0x03) {
            const us = (buf[i + 3] << 16) | (buf[i + 4] << 8) | buf[i + 5]
            if (us > 0) bpm = Math.round((60_000_000 / us) * 10) / 10
            break
          }
        }
        resolve({ bpm })
      } catch {
        resolve({ bpm: 120 })
      }
    }
    reader.onerror = () => resolve({ bpm: 120 })
    reader.readAsArrayBuffer(file)
  })
}

const downloadModel = async (lang: string) => {
  downloadingLangs.value.push(lang)
  try {
    const res = await fetch(`/api/mfa/download-model/${lang}`, { method: 'POST' })
    const data = await res.json()
    if (data.success) {
      ElMessage.success(t('processor.modelDownloaded', { lang: lang.toUpperCase() }))
      await checkSystemStatus()
    } else {
      ElMessage.error(t('processor.modelDownloadFailed', { error: data.error }))
    }
  } catch (e) {
    ElMessage.error(t('processor.modelDownloadError', { error: String(e) }))
  } finally {
    downloadingLangs.value = downloadingLangs.value.filter(l => l !== lang)
  }
}

// 核心核心控制逻辑：开始处理
const processAudio = async () => {
  // ============================================================
  // 分支 1) 仅工程文件模式：WAV + LAB -> 直接转工程文件
  // ============================================================
if (processingMode.value === 'project-only') {
  if (!formData.value.audioFile) {
    ElMessage.warning(t('processor.selectWav'))
    return
  }

  const notationFile = selectedNotationFile.value
  if (!notationFile) {
    ElMessage.warning(t('processor.selectLabOrMidi'))
    return
  }

  const notationExt = notationFile.name.toLowerCase().split('.').pop() || ''
  if (!['lab', 'mid', 'midi'].includes(notationExt)) {
    ElMessage.warning(t('processor.selectValidNotation'))
    return
  }

  clearJobPolling()
  processing.value = true
  progressPercent.value = 0
  error.value = ''
  result.value = null
  currentJobId.value = ''
  resetProcessingSteps()
  updateProcessingStep(0, t('processor.statusSkipped'), t('processor.projectModeSkipAlign'))
  updateProcessingStep(1, t('processor.statusProcessing'), t('processor.projectModeProcessing'))
  updateProcessingStep(2, t('processor.statusWaiting'), t('processor.projectModeWaitProject'))

  let progressTimer: number | null = null

  try {
    const formDataObj = new FormData()
    formDataObj.append('wav_file', formData.value.audioFile)
    formDataObj.append('format', formData.value.outputFormat)
    if (formData.value.outputFormat === 'vsqx') {
      formDataObj.append('vsqx_singer',    vsqxSingerConfig.value.name)
      formDataObj.append('vsqx_singer_id', vsqxSingerConfig.value.id)
    }
    formDataObj.append('title', formData.value.projectTitle)
    formDataObj.append('phoneme_mode', formData.value.phonemeMode)
    formDataObj.append('bpm', advancedConfig.value.bpm.toString())
    formDataObj.append('base_pitch', advancedConfig.value.base_pitch.toString())
    formDataObj.append('f0_method', advancedConfig.value.f0_method)
    formDataObj.append('f0_device', advancedConfig.value.f0_device)
    formDataObj.append('crepe_model', advancedConfig.value.crepe_model)
    formDataObj.append('f0_smooth', advancedConfig.value.f0_smooth.toString())
    formDataObj.append('f0_smooth_window', advancedConfig.value.f0_smooth_window.toString())
    formDataObj.append('precision', advancedConfig.value.precision)
    formDataObj.append('f0_floor', advancedConfig.value.f0_floor.toString())
    formDataObj.append('f0_ceil', advancedConfig.value.f0_ceil.toString())
    formDataObj.append('auto_note_pitch', advancedConfig.value.auto_note_pitch.toString())
    formDataObj.append('export_pitch_line', advancedConfig.value.export_pitch_line.toString())
    // 【历史 bug 修复】project-only 模式此前写死传 'false'，导致开关形同虚设；
    // project-only 不存在对齐环节，不受 englishWordAlign 约束，只看开关本身 + 输出格式。
    formDataObj.append('word_phoneme_map', (
      wordPhonemeMap.value &&
      (formData.value.outputFormat === 'sv' || formData.value.outputFormat === 'vsqx')
    ).toString())
    formDataObj.append('dict_source', dictSource.value)

    // 只传一个标注文件：LAB 或 MIDI 二选一
    if (notationExt === 'lab') {
      formDataObj.append('lab_file', notationFile)
    } else {
      formDataObj.append('midi_file', notationFile)
    }

    progressTimer = window.setInterval(() => {
      if (progressPercent.value < 30) progressPercent.value += 3
    }, 400)

    const res = await fetch('/api/pipeline/project-only', {
      method: 'POST',
      body: formDataObj,
    })
    const data = await res.json()

    if (!res.ok) throw new Error(data.error || t('processor.submitFailed'))

    if (data.job_id) {
      if (progressTimer !== null) { window.clearInterval(progressTimer); progressTimer = null }
      progressPercent.value = 35

      const finalPayload = await waitForJobFinished(data.job_id)
      const normalized = normalizeResult(finalPayload)

      if (!normalized.projectPath) throw new Error(t('processor.projectMissing'))

      result.value = normalized
      progressPercent.value = 100
      updateProcessingStep(0, t('processor.statusSkipped'), t('processor.projectModeNoAlign'))
      updateProcessingStep(1, t('processor.statusDone'), t('processor.projectModeF0Done'))
      updateProcessingStep(2, t('processor.statusDone'), `${t('processor.projectFile')}: ${getFileName(normalized.projectPath)}`)
      ElMessage.success(`✅ ${t('processor.projectModeSuccess')}`)
      return
    }

    if (!data.success) throw new Error(data.error || t('processor.submitFailed'))
    const normalized = normalizeResult(data)
    result.value = normalized
    progressPercent.value = 100
    updateProcessingStep(0, t('processor.statusSkipped'), t('processor.projectModeNoAlign'))
    updateProcessingStep(1, t('processor.statusDone'), t('processor.projectModeF0Done'))
    updateProcessingStep(2, t('processor.statusDone'), `${t('processor.projectFile')}: ${getFileName(normalized.projectPath || '')}`)
    ElMessage.success(`✅ ${t('processor.projectModeSuccess')}`)
  } catch (e: any) {
    error.value = e?.message || String(e)
    ElMessage.error(`❌ ${error.value}`)
  } finally {
    if (progressTimer !== null) window.clearInterval(progressTimer)
    clearJobPolling()
    processing.value = false
  }
  return
}

  // ============================================================
  // 分支 2) 其他传统模式：需要音频，非 ASR 后端需要文本
  // ============================================================
  if (!formData.value.audioFile) {
    ElMessage.warning(t('processor.selectAudio'))
    return
  }
  if (!formData.value.text.trim() && !isTextOptional.value) {
    ElMessage.warning(t('processor.selectText'))
    return
  }
  if (!isReady.value) {
    ElMessage.error(t('processor.backendNotReady'))
    return
  }

  const maxSize = 512 * 1024 * 1024
  if (formData.value.audioFile.size > maxSize) {
    ElMessage.warning(t('processor.fileTooLarge'))
  }

  clearJobPolling()
  processing.value = true
  progressPercent.value = 0
  error.value = ''
  result.value = null
  currentJobId.value = ''
  resetProcessingSteps()

  let progressTimer: number | null = null

  try {
    const formDataObj = new FormData()
    formDataObj.append('audio_file', formData.value.audioFile)
    formDataObj.append('text', formData.value.text)
    formDataObj.append('language', formData.value.language)
    formDataObj.append('aligner_backend', alignerBackend.value)
    formDataObj.append('aligner_device', advancedConfig.value.aligner_device)
    formDataObj.append('whisperx_model', advancedConfig.value.whisperx_model)
    formDataObj.append('english_word_align', (englishWordAlign.value && formData.value.language !== 'jpn').toString())

    if (processingMode.value === 'full') {
      formDataObj.append('format', formData.value.outputFormat)
      if (formData.value.outputFormat === 'vsqx') {
        formDataObj.append('vsqx_singer',    vsqxSingerConfig.value.name)
        formDataObj.append('vsqx_singer_id', vsqxSingerConfig.value.id)
      }
      formDataObj.append('title', formData.value.projectTitle)
      formDataObj.append('bpm', advancedConfig.value.bpm.toString())
      formDataObj.append('base_pitch', advancedConfig.value.base_pitch.toString())
      formDataObj.append('f0_method', advancedConfig.value.f0_method)
      formDataObj.append('f0_device', advancedConfig.value.f0_device)
      formDataObj.append('crepe_model', advancedConfig.value.crepe_model)
      formDataObj.append('f0_smooth', advancedConfig.value.f0_smooth.toString())
      formDataObj.append('f0_smooth_window', advancedConfig.value.f0_smooth_window.toString())
      formDataObj.append('precision', advancedConfig.value.precision)
      formDataObj.append('f0_floor', advancedConfig.value.f0_floor.toString())
      formDataObj.append('f0_ceil', advancedConfig.value.f0_ceil.toString())
      formDataObj.append('auto_note_pitch', advancedConfig.value.auto_note_pitch.toString())
      formDataObj.append('export_pitch_line', advancedConfig.value.export_pitch_line.toString())
      formDataObj.append('word_phoneme_map', (
        wordPhonemeMap.value &&
        englishWordAlign.value &&
        (formData.value.outputFormat === 'sv' || formData.value.outputFormat === 'vsqx')
      ).toString())
      formDataObj.append('dict_source', dictSource.value)
    }

    progressTimer = window.setInterval(() => {
      if (progressPercent.value < 30) progressPercent.value += 3
    }, 400)

    const endpoint = processingMode.value === 'full' ? '/api/pipeline/full' : '/api/pipeline/mfa-only'
    const res = await fetch(endpoint, { method: 'POST', body: formDataObj })
    const data = await res.json()

    if (!res.ok) throw new Error(data.error || t('processor.submitFailed'))

    // full 和 mfa-only 均走异步轮询（后端返回 job_id）
    if (data.job_id) {
      if (progressTimer !== null) { window.clearInterval(progressTimer); progressTimer = null }
      progressPercent.value = 35

      if (processingMode.value === 'mfa-only') {
        updateProcessingStep(0, t('processor.statusProcessing'), `${alignerBackendLabel.value} ${t('processor.processing')}...`)
        updateProcessingStep(1, t('processor.statusWaiting'), t('processor.projectModeNoAlign'))
        updateProcessingStep(2, t('processor.statusWaiting'), t('processor.projectModeNoAlign'))
      } else {
        updateProcessingStep(0, t('processor.statusProcessing'), `${alignerBackendLabel.value} ${t('processor.projectModeProcessing')}`)
        updateProcessingStep(1, t('processor.statusWaiting'), t('processor.stageExtractF0'))
        updateProcessingStep(2, t('processor.statusWaiting'), t('processor.projectModeWaitProject'))
      }

      const finalPayload = await waitForJobFinished(data.job_id)
      const normalized = normalizeResult(finalPayload)

      if (processingMode.value === 'full') {
        if (!normalized.projectPath) throw new Error(t('processor.projectMissing'))
        updateProcessingStep(0, t('processor.statusDone'), `${t('processor.backendMfa')} ${t('processor.statusDone')}`)
        updateProcessingStep(1, t('processor.statusDone'), t('processor.projectModeF0Done'))
        updateProcessingStep(2, t('processor.statusDone'), `${t('processor.projectFile')}: ${getFileName(normalized.projectPath)}`)
      } else {
        // mfa-only
        if (!normalized.labContent) throw new Error(t('processor.labEmpty'))
        const segCount = countLabSegments(normalized.labContent)
        updateProcessingStep(0, t('processor.statusDone'), `${segCount} ${t('processor.segmentCount')}`)
        updateProcessingStep(1, t('processor.statusSkipped'), t('processor.projectModeNoAlign'))
        updateProcessingStep(2, t('processor.statusSkipped'), t('processor.projectModeNoAlign'))
      }

      result.value = normalized
      progressPercent.value = 100
      ElMessage.success(`✅ ${t('processor.success')}`)
      return
    }

    // 向下兼容：full 模式同步结果回退
    if (processingMode.value === 'full') {
      const normalized = normalizeResult(data)
      if (data.success && normalized.projectPath) {
        updateProcessingStep(0, t('processor.statusDone'), `${t('processor.backendMfa')} ${t('processor.statusDone')}`)
        updateProcessingStep(1, t('processor.statusDone'), t('processor.projectModeF0Done'))
        updateProcessingStep(2, t('processor.statusDone'), `${t('processor.projectFile')}: ${getFileName(normalized.projectPath)}`)
        result.value = normalized
        progressPercent.value = 100
        ElMessage.success(`✅ ${t('processor.success')}`)
        return
      }
      throw new Error(data.error || t('processor.projectMissing'))
    }

    // mfa-only 同步回退（后端已异步，此分支仅做兼容保留）
    if (!data.success) throw new Error(data.error || t('processor.jobFailed'))
    const normalized = normalizeResult(data)
    if (!normalized.labContent) throw new Error(t('processor.labEmpty'))
    const segCount = countLabSegments(normalized.labContent)
    result.value = normalized
    progressPercent.value = 100
    updateProcessingStep(0, t('processor.statusDone'), `${segCount} ${t('processor.segmentCount')}`)
    updateProcessingStep(1, t('processor.statusSkipped'), t('processor.projectModeNoAlign'))
    updateProcessingStep(2, t('processor.statusSkipped'), t('processor.projectModeNoAlign'))
    ElMessage.success(`✅ ${t('processor.success')}`)
  } catch (e: any) {
    error.value = e?.message || String(e)
    ElMessage.error(`❌ ${error.value}`)
  } finally {
    if (progressTimer !== null) window.clearInterval(progressTimer)
    clearJobPolling()
    processing.value = false
  }
}

const downloadLab = () => {
  if (!result.value?.labContent) {
    ElMessage.warning(t('processor.noLabContent'))
    return
  }

  // 智能获取与工程文件一致的 stem（包含随机后缀）
  let stem = 'alignment'

  if (result.value.projectPath) {
    const projName = getFileName(result.value.projectPath)
    stem = projName.replace(/\.(svp|ustx|sv|vsqx)$/, '')   // 去掉扩展名
  } else if (result.value.labPath) {
    const labName = getFileName(result.value.labPath)
    stem = labName.replace(/\.lab$/, '')
  } else if (formData.value.audioFile) {
    stem = formData.value.audioFile.name.replace(/\.\w+$/, '')
  }

  const filename = `${stem}.lab`

  const element = document.createElement('a')
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(result.value.labContent))
  element.setAttribute('download', filename)
  document.body.appendChild(element)
  element.click()
  document.body.removeChild(element)

  ElMessage.success(`✅ ${t('processor.downloadLabFile')}: ${filename}`)
}

const downloadProject = async () => {
  if (!result.value?.projectPath) return
  downloadingProject.value = true
  try {
    const filename = result.value.projectPath.split(/[\\/]/).pop()
    const response = await fetch(`/api/work-dir/download/${encodeURIComponent(filename)}`)
    if (!response.ok) {
      ElMessage.error(t('processor.submitFailed'))
      return
    }
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const element = document.createElement('a')
    element.href = url
    element.download = filename
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
    window.URL.revokeObjectURL(url)
    ElMessage.success(t('processor.downloadProjectFile'))
  } catch (e) {
    ElMessage.error(`${t('processor.submitFailed')}: ${e}`)
  } finally {
    downloadingProject.value = false
  }
}

const copyLabToClipboard = () => {
  if (!result.value?.labContent) return
  navigator.clipboard.writeText(result.value.labContent).then(() => {
    ElMessage.success(t('processor.copied'))
  }).catch(() => {
    ElMessage.error(t('processor.copyFailed'))
  })
}

const reset = () => {
  clearJobPolling()
  formData.value = {
    audioFile: null,
    labFile: null,
    midiFile: null,
    text: '',
    language: 'cmn',
    outputFormat: 'sv',
    projectTitle: t('processor.defaultProjectTitle'),
    phonemeMode: 'none'
  }
  midiInfo.value = { bpm: 120, loaded: false }
  labMidiUploadKey.value += 1
  audioUploadKey.value += 1
  result.value = null
  error.value = ''
  progressPercent.value = 0
  currentJobId.value = ''
  resetProcessingSteps()
  // alignerBackend 保留用户选择，不重置
}

const newProcess = () => {
  reset()
}
</script>

<style scoped>
/* 样式部分保持不变 */
.processor-container {
  width: 100%;
}

.dict-source-hint {
  margin-top: 6px;
  font-size: 12px;
  color: #909399;
  line-height: 1.5;
}

.processor-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

.processor-form :deep(.el-form-item) {
  margin-bottom: 20px;
}

.processor-form :deep(.el-form-item__label) {
  white-space: normal;
  line-height: 1.35;
  padding-bottom: 8px;
}

.processor-form :deep(.el-form-item__content) {
  width: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.card-title {
  font-size: 16px;
  font-weight: bold;
  color: #333;
}

.header-actions {
  display: flex;
  gap: 15px;
  align-items: center;
}

.file-info {
  margin-top: 10px;
  padding: 8px 12px;
  background: #ecf5ff;
  color: #409eff;
  border-radius: 4px;
  font-size: 12px;
}

.help-text {
  color: #909399;
  font-size: 12px;
  margin-top: 5px;
}

.help-text .text-optional-hint {
  color: #67c23a;
  font-weight: 500;
}

/* 确保模式帮助文本在 Element 表单条目中强制换行，不向右侧外溢 */
.mode-help {
  width: 100%;
  display: block;
  color: #909399;
  font-size: 12px;
  margin-top: 6px;
}

.option-hint {
  margin-left: 10px;
  color: #909399;
  font-size: 12px;
  line-height: 1.5;
}

.option-hint code {
  background: #f0f0f0;
  border-radius: 3px;
  padding: 0 3px;
  font-family: monospace;
  color: #476582;
}

.pitch-input-group {
  display: flex;
  gap: 10px;
  align-items: center;
}

.pitch-name {
  color: #409eff;
  font-weight: bold;
  font-size: 14px;
  min-width: 50px;
}

/* MIDI 相关样式 */
.midi-loaded {
  background: #f0f9eb !important;
  color: #67c23a !important;
  border: 1px solid #c2e7b0;
}

.midi-bpm-tag {
  display: inline-block;
  margin-left: 10px;
  padding: 1px 8px;
  background: #67c23a;
  color: white;
  border-radius: 10px;
  font-size: 11px;
  font-weight: bold;
  letter-spacing: 0.5px;
}

.midi-lock-tip {
  color: #909399;
  font-size: 11px;
  margin-left: 8px;
}

.icon-tip {
  margin-left: 5px;
  color: #909399;
}

.settings-info {
  margin-top: 15px;
}

.settings-info p {
  margin: 8px 0;
  font-size: 12px;
}

.settings-info strong {
  color: #333;
}

.progress-bar {
  margin-top: 15px;
}

.disabled-text {
  color: #f56c6c;
  font-size: 12px;
  margin-left: 10px;
}

.result-section {
  margin-top: 30px;
  padding-top: 20px;
}

.result-info {
  margin-bottom: 15px;
  padding: 15px;
  background: #f0f9ff;
  border-radius: 4px;
  border-left: 4px solid #409eff;
}

.result-info p {
  margin: 8px 0;
  color: #606266;
  font-size: 12px;
}

.result-info code {
  background: #fff;
  padding: 2px 6px;
  border-radius: 2px;
  font-family: 'Courier New', monospace;
}

.output-text {
  font-family: 'Courier New', monospace;
  font-size: 11px;
  margin: 15px 0;
}

.tab-actions {
  margin-top: 10px;
  display: flex;
  gap: 10px;
}

.file-info-box {
  padding: 15px;
  background: #f5f5f5;
  border-radius: 4px;
}

.file-info-box p {
  margin: 12px 0 6px;
  font-weight: bold;
  color: #333;
}

.file-info-box code {
  display: block;
  background: white;
  padding: 8px;
  border-radius: 4px;
  margin: 0 0 12px;
  word-break: break-all;
  font-family: 'Courier New', monospace;
  font-size: 11px;
  border: 1px solid #dcdfe6;
}

.file-info-box ul {
  margin: 8px 0 0 20px;
  font-size: 12px;
  color: #606266;
}

.file-info-box li {
  margin: 4px 0;
}

.details-box {
  padding: 15px 0;
}

.action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  flex-wrap: wrap;
}

.action-buttons :deep(.el-button) {
  flex: 1;
  min-width: 150px;
}

.error-section {
  margin-top: 20px;
}

.status-box {
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-top: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.status-item {
  margin-bottom: 15px;
}

.status-item .label {
  display: block;
  color: #606266;
  font-weight: bold;
  margin-bottom: 5px;
}

.model-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px;
  background: #f5f5f5;
  border-radius: 4px;
}

.warning-box {
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-top: 20px;
  box-shadow: 0 2px 12px rgba(255, 177, 0, 0.2);
}

.warning-box code {
  background: #f5f5f5;
  padding: 8px 12px;
  border-radius: 4px;
  display: block;
  margin: 10px 0;
  font-family: 'Courier New', monospace;
  color: #d63200;
  font-size: 12px;
}

@media (max-width: 768px) {
  .card-header {
    flex-direction: column;
    gap: 15px;
  }

  .action-buttons {
    flex-direction: column;
  }

  .action-buttons :deep(.el-button) {
    width: 100%;
  }

  .pitch-input-group {
    flex-direction: column;
  }
}
</style>

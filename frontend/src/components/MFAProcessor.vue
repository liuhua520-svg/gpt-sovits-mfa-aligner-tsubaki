<template>
  <div class="processor-container">
    <el-card class="processor-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span class="card-title">📁 单文件处理</span>
          <div class="header-actions">
            <el-tooltip content="访问GitHub项目" placement="bottom">
              <el-button 
                link 
                @click="openGitHub"
                type="primary"
              >
                🔗 GitHub 项目链接
              </el-button>
            </el-tooltip>
            <el-tooltip content="检查状态" placement="bottom">
              <el-button link @click="refreshStatus" :loading="checkingStatus">
                🔄 检查状态
              </el-button>
            </el-tooltip>
          </div>
        </div>
      </template>

      <el-form :model="formData" label-width="100px">
        <el-form-item label="音频文件">
          <el-upload
            drag
            action="#"
            :auto-upload="false"
            :limit="1"
            :on-exceed="handleExceed"
            @change="handleAudioSelect"
            accept=".wav,.mp3,.flac,.m4a,.aac"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              拖拽或<em>点击选择</em>音频文件
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持 WAV / MP3 / FLAC / M4A / AAC，最大 512MB
              </div>
            </template>
          </el-upload>
          <div v-if="formData.audioFile" class="file-info">
            ✓ {{ formData.audioFile.name }} ({{ formatFileSize(formData.audioFile.size) }})
          </div>
        </el-form-item>

        <!-- LAB / MIDI 合并上传（仅 project-only 模式） -->
        <el-form-item v-if="processingMode === 'project-only'" label="LAB / MIDI 文件">
          <el-upload
            :key="labMidiUploadKey"
            drag
            action="#"
            :auto-upload="false"
            :multiple="true"
            :limit="2"
            :on-exceed="handleLabMidiExceed"
            @change="handleLabMidiChange"
            accept=".lab,.mid,.midi"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              拖拽或<em>点击选择</em>文件（可同时选择 LAB + MIDI）
            </div>
            <template #tip>
              <div class="el-upload__tip">
                <strong>.lab</strong>（必须）标注文件
                &nbsp;+&nbsp;
                <strong>.mid / .midi</strong>（可选）导入后音符音高与 BPM 从 MIDI 读取
              </div>
            </template>
          </el-upload>

          <!-- 已选文件状态行 -->
          <div v-if="formData.labFile" class="file-info" style="margin-top:6px">
            📄 {{ formData.labFile.name }} ({{ formatFileSize(formData.labFile.size) }})
          </div>
          <div v-if="formData.midiFile" class="file-info midi-loaded" style="margin-top:4px">
            🎹 {{ formData.midiFile.name }} ({{ formatFileSize(formData.midiFile.size) }})
            <span v-if="midiInfo.loaded" class="midi-bpm-tag">BPM {{ midiInfo.bpm }}</span>
          </div>

          <!-- MIDI 接管提示 -->
          <el-alert
            v-if="midiLoaded"
            type="info"
            :closable="false"
            show-icon
            style="margin-top:8px"
          >
            <template #title>已导入 MIDI — 以下选项已由 MIDI 数据接管</template>
            <p style="margin:4px 0 0;font-size:12px;color:#606266">
              🔒 自动音符音高 &nbsp;·&nbsp; BPM &nbsp;·&nbsp; 基准音高 (MIDI Note)
              <br>将直接从 MIDI 文件中读取，手动调整已被禁用
            </p>
          </el-alert>
        </el-form-item>

        <el-form-item v-if="processingMode !== 'project-only'" label="输入文本">
          <el-input
            v-model="formData.text"
            type="textarea"
            :rows="4"
            placeholder="粘贴文本内容"
            show-word-limit
          />
          <div class="help-text">
            当前字符数：{{ formData.text.length }}
          </div>
        </el-form-item>

        <el-form-item v-if="processingMode !== 'project-only'" label="语言">
          <el-select v-model="formData.language" placeholder="选择语言">
            <el-option label="普通话 🇨🇳" value="cmn" />
            <el-option label="英语 🇬🇧" value="eng" />
            <el-option label="日语 🇯🇵" value="jpn" />
            <el-option label="韩语 🇰🇷" value="kor" />
            <el-option label="粤语 🇭🇰" value="yue" />
          </el-select>
        </el-form-item>

        <el-form-item label="处理模式">
          <el-radio-group v-model="processingMode">
            <el-radio value="mfa-only">仅标注 (快速)</el-radio>
            <el-radio value="full">完整处理 (标注+F0+工程文件)</el-radio>
            <el-radio value="project-only">仅生成工程 (WAV + LAB)</el-radio>
          </el-radio-group>
          <div class="mode-help">
            <small v-if="processingMode === 'mfa-only'">
              只进行 MFA 自动标注，生成 LAB 文件
            </small>
            <small v-else-if="processingMode === 'full'">
              执行完整流程：标注 → F0提取 → 工程文件生成
            </small>
            <small v-else>
              直接整合现有 WAV 和 LAB 文件生成工程文件，跳过 MFA 自动标注
            </small>
          </div>
        </el-form-item>

        <el-form-item v-if="processingMode === 'full' || processingMode === 'project-only'" label="输出格式">
          <el-select v-model="formData.outputFormat" placeholder="选择输出格式">
            <el-option label="Synthesizer V Studio (.svp)" value="sv" />
            <el-option label="OpenUtau/UTAU (.ustx)" value="utau" />
          </el-select>
        </el-form-item>

        <el-form-item v-if="processingMode === 'project-only'" label="音素转换">
          <el-radio-group v-model="formData.phonemeMode">
            <el-radio value="none">不转换</el-radio>
            <el-radio value="merge">合并辅音</el-radio>
            <el-radio value="hiragana">平假名</el-radio>
            <el-radio value="katakana">片假名</el-radio>
          </el-radio-group>
          <div class="help-text">
            <small v-if="formData.phonemeMode === 'none'">
              保持 LAB 中的原始音素标签不变（适用于所有语言）
            </small>
            <small v-else-if="formData.phonemeMode === 'merge'">
              将辅音与后续元音合并为罗马字音节（s + a → sa，N → N，p + u → pu）
            </small>
            <small v-else-if="formData.phonemeMode === 'hiragana'">
              合并辅音+元音并转换为平假名（s + a → さ，N → ん，p + u → ぷ）
            </small>
            <small v-else>
              合并辅音+元音并转换为片假名（s + a → サ，N → ン，p + u → プ）
            </small>
          </div>
          <div class="help-text" style="margin-top:4px">
            <small style="color:#909399">
              ⚠ 合并/假名转换适用于含逐个音素的日语 LAB 文件（如原始 MFA 输出）
            </small>
          </div>
        </el-form-item>

        <el-form-item v-if="processingMode === 'full' || processingMode === 'project-only'" label="音轨名">
          <el-input
            v-model="formData.projectTitle"
            placeholder="输入音轨名"
            maxlength="248"
          />
        </el-form-item>

        <el-collapse v-if="processingMode === 'full' || processingMode === 'project-only'" accordion>
          <el-collapse-item title="⚙️ 高级设置" name="advanced">
            <el-row :gutter="20">
              <el-col :xs="24" :sm="12">
                <el-form-item label="BPM (节拍/分钟)">
                  <el-input-number
                    v-model="advancedConfig.bpm"
                    :min="20"
                    :max="300"
                    :step="1"
                    controls-position="right"
                    :disabled="midiLoaded"
                  />
                  <span v-if="midiLoaded && midiInfo.loaded" class="midi-lock-tip">
                    🔒 {{ midiInfo.bpm }} (来自 MIDI)
                  </span>
                  <span v-else-if="midiLoaded" class="midi-lock-tip">
                    🔒 从 MIDI 读取
                  </span>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="基准音高 (MIDI Note)">
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
                    <span v-if="midiLoaded" class="midi-lock-tip">🔒 从 MIDI 读取</span>
                  </div>
                </el-form-item>
              </el-col>

              <el-col :xs="24">
                <el-divider>📈 音高精细控制</el-divider>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="自动音符音高">
                  <el-switch 
                    v-model="advancedConfig.auto_note_pitch"
                    active-text="自动对齐实际音高"
                    inactive-text="固定在基准音高"
                    :disabled="midiLoaded"
                  />
                  <span v-if="midiLoaded" class="midi-lock-tip" style="display:block;margin-top:4px">
                    🔒 音符音高由 MIDI 提供
                  </span>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="导出连续音高">
                  <el-switch 
                    v-model="advancedConfig.export_pitch_line"
                    active-text="写入 F0 曲线参数"
                    inactive-text="仅生成纯净音符"
                  />
                </el-form-item>
              </el-col>

              <el-col :xs="24">
                <el-divider>F0 提取算法与范围</el-divider>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="F0 提取方法">
                  <el-radio-group v-model="advancedConfig.f0_method" :disabled="!advancedConfig.export_pitch_line && !advancedConfig.auto_note_pitch">
                    <el-radio label="dio">
                      <span>DIO (快速)</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                    <el-radio label="harvest">
                      <span>Harvest (精确)</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                    <el-radio label="crepe" :disabled="systemStatus.audio_processing?.f0_backends?.crepe?.available === false">
                      <span>CREPE (神经网络，抗噪)</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                    <el-radio label="rmvpe" :disabled="systemStatus.audio_processing?.f0_backends?.rmvpe?.available === false">
                      <span>RMVPE (深度模型，最鲁棒)</span>
                      <el-icon class="icon-tip"><InfoFilled /></el-icon>
                    </el-radio>
                  </el-radio-group>
                  <p v-if="advancedConfig.f0_method === 'crepe' && systemStatus.audio_processing?.f0_backends?.crepe?.available === false" class="help-text">
                    ⚠ 未检测到 torch / torchcrepe，请先安装依赖
                  </p>
                  <p v-if="advancedConfig.f0_method === 'rmvpe' && systemStatus.audio_processing?.f0_backends?.rmvpe?.available === false" class="help-text">
                    ⚠ 未检测到 RMVPE 模型权重，请下载 rmvpe.pt 并放入 models/rmvpe/ 目录
                  </p>
                </el-form-item>
              </el-col>

              <el-col v-if="advancedConfig.f0_method === 'crepe'" :xs="24" :sm="12">
                <el-form-item label="CREPE 模型规格">
                  <el-radio-group v-model="advancedConfig.crepe_model">
                    <el-radio label="full">Full (精度高)</el-radio>
                    <el-radio label="tiny">Tiny (速度快)</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>

              <el-col v-if="advancedConfig.f0_method === 'crepe' || advancedConfig.f0_method === 'rmvpe'" :xs="24" :sm="12">
                <el-form-item label="运行设备">
                  <el-radio-group v-model="advancedConfig.f0_device">
                    <el-radio label="auto">自动 (优先 GPU)</el-radio>
                    <el-radio label="cpu">CPU</el-radio>
                    <el-radio label="cuda">CUDA (GPU)</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="浮点精度">
                  <el-radio-group v-model="advancedConfig.precision">
                    <el-radio label="single">单精度 (Float32)</el-radio>
                    <el-radio label="double">双精度 (Float64)</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="F0 平滑处理">
                  <el-switch 
                    v-model="advancedConfig.f0_smooth"
                    active-text="启用"
                    inactive-text="禁用"
                    :disabled="!advancedConfig.export_pitch_line"
                  />
                </el-form-item>
              </el-col>

              <el-col v-if="advancedConfig.f0_smooth" :xs="24" :sm="12">
                <el-form-item label="平滑窗口大小">
                  <el-input-number
                    v-model="advancedConfig.f0_smooth_window"
                    :min="1"
                    :max="21"
                    :step="2"
                    controls-position="right"
                    :disabled="!advancedConfig.export_pitch_line"
                  />
                  <span class="help-text">推荐值: 3-7 (越大越平滑)</span>
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="最低频率 (Hz)">
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
                <el-form-item label="最高频率 (Hz)">
                  <el-input-number
                    v-model="advancedConfig.f0_ceil"
                    :min="300"
                    :max="1000"
                    :step="50"
                    controls-position="right"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-alert type="info" :closable="false" show-icon class="settings-info">
              <template #title>💡 高级设置说明</template>
              <p><strong>BPM:</strong> 用于工程文件的节拍数，不影响实际处理速度</p>
              <p><strong>基准音高:</strong> 工程文件的默认基准音（60 = C4）</p>
              <p><strong>自动音符音高:</strong> 音符块将自动放置在音频识别出的实际 MIDI 键位上</p>
              <p><strong>导出连续音高:</strong> 是否将精细的基频起伏数据（F0）以参数线形式导入到工程底部</p>
              <p><strong>DIO / Harvest / CREPE / RMVPE：</strong>DIO 最快但偶有跳音；Harvest 更精确但稍慢；CREPE 基于神经网络，对噪声/伴奏有较强鲁棒性；RMVPE 是目前对人声最鲁棒的深度模型，推荐在条件允许时优先使用（CREPE/RMVPE 需要 torch，RMVPE 还需额外下载模型权重）</p>
              <p><strong>运行设备：</strong>CREPE/RMVPE 可选择 CPU 或 CUDA(GPU)，GPU 可大幅加速；"自动"会在检测到可用 GPU 时优先使用</p>
              <p><strong>F0 范围:</strong> 女声通常 150-300Hz，男声 80-150Hz，根据需要调整</p>
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
            <span v-if="!processing">🚀 开始处理</span>
            <span v-else>处理中... {{ progressPercent }}%</span>
          </el-button>
          <el-button @click="reset" :disabled="processing">🔄 重置</el-button>
          <span v-if="!isReady && processingMode !== 'project-only'" class="disabled-text">
            (系统未就绪或语言模型未下载)
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

        <h3>✅ 处理结果</h3>
        <div class="result-info">
          <el-row :gutter="20">
            <el-col :xs="24" :sm="12">
              <p><strong>处理时间:</strong> {{ formatTime(result.processingTime) }}</p>
              <p v-if="result.labPath"><strong>LAB 文件:</strong> {{ getFileName(result.labPath) }}</p>
            </el-col>
            <el-col :xs="24" :sm="12">
              <p v-if="result.projectPath"><strong>工程文件:</strong> {{ getFileName(result.projectPath) }}</p>
              <p v-if="result.segments"><strong>标注段数:</strong> {{ result.segments }}</p>
            </el-col>
          </el-row>
        </div>

        <el-tabs>
          <el-tab-pane v-if="result.labContent" label="LAB 标注内容">
            <el-input
              v-model="result.labContent"
              type="textarea"
              :rows="12"
              readonly
              class="output-text"
            />
            <div class="tab-actions">
              <el-button @click="copyLabToClipboard" size="small">
                📋 复制 LAB
              </el-button>
              <el-button @click="downloadLab" size="small" type="success">
                📥 下载 LAB
              </el-button>
            </div>
          </el-tab-pane>

          <el-tab-pane v-if="result.projectPath" label="文件信息">
            <div class="file-info-box">
              <el-row :gutter="20">
                <el-col :xs="24" v-if="result.labPath">
                  <p><strong>LAB 标注文件:</strong></p>
                  <code>{{ result.labPath }}</code>
                </el-col>
                <el-col :xs="24">
                  <p><strong>工程文件:</strong></p>
                  <code>{{ result.projectPath }}</code>
                </el-col>
                <el-col :xs="24">
                  <p><strong>输出格式:</strong> {{ result.projectFormat === 'sv' ? 'Synthesizer V Studio' : 'OpenUtau/UTAU' }}</p>
                  <p v-if="result.segments"><strong>标注段数:</strong> {{ result.segments }}</p>
                  <p v-if="result.config"><strong>处理配置:</strong></p>
                  <ul v-if="result.config">
                    <li>BPM: {{ result.config.bpm }}</li>
                    <li>基准音高: {{ midiNoteToName(result.config.base_pitch) }} (MIDI {{ result.config.base_pitch }})</li>
                    <li>自动音符音高: {{ result.config.auto_note_pitch ? '已启用' : '已禁用' }}</li>
                    <li>导出连续音高: {{ result.config.export_pitch_line ? '已启用' : '已禁用' }}</li>
                    <li>F0 方法: {{ result.config.f0_method?.toUpperCase?.() || result.config.f0_method }}</li>
                    <li v-if="result.config.f0_method === 'crepe'">CREPE 模型: {{ result.config.crepe_model }}</li>
                    <li v-if="result.config.f0_method === 'crepe' || result.config.f0_method === 'rmvpe'">运行设备: {{ result.config.f0_device }}</li>
                    <li>精度: {{ result.config.use_double_precision ? '双精度 (Float64)' : '单精度 (Float32)' }}</li>
                  </ul>
                </el-col>
              </el-row>
            </div>
          </el-tab-pane>

          <el-tab-pane label="处理详情">
            <div class="details-box">
              <el-table :data="processingDetails" stripe style="width: 100%">
                <el-table-column prop="stage" label="处理阶段" width="200" />
                <el-table-column prop="status" label="状态" width="100">
                  <template #default="{ row }">
                    <el-tag v-if="row.status === '完成'" type="success">{{ row.status }}</el-tag>
                    <el-tag v-else-if="row.status === '跳过'" type="warning">{{ row.status }}</el-tag>
                    <el-tag v-else type="info">{{ row.status }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="message" label="详情" show-overflow-tooltip />
              </el-table>
            </div>
          </el-tab-pane>
        </el-tabs>

        <div class="action-buttons">
          <el-button v-if="result.labContent" type="success" @click="downloadLab" size="large">
            📥 下载 LAB 文件
          </el-button>
          <el-button 
            v-if="result.projectPath" 
            type="success" 
            @click="downloadProject" 
            size="large"
            :loading="downloadingProject"
          >
            📥 下载工程文件
          </el-button>
          <el-button v-if="result.labContent" @click="copyLabToClipboard" size="large">
            📋 复制 LAB 内容
          </el-button>
          <el-button type="info" @click="newProcess" size="large">
            🔄 处理下一个
          </el-button>
        </div>
      </div>

      <div v-if="error" class="error-section">
        <el-alert
          :title="`错误: ${error}`"
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
          <span>🔧 系统状态</span>
        </template>

        <el-row :gutter="20">
          <el-col :xs="24" :sm="12">
            <div class="status-item">
              <span class="label">MFA 状态:</span>
              <el-tag :type="systemStatus.mfa?.installed ? 'success' : 'danger'" size="large">
                {{ systemStatus.mfa?.installed ? '✓ 已安装' : '✗ 未安装' }}
              </el-tag>
            </div>
            <div v-if="systemStatus.mfa?.installed" class="status-item">
              <span class="label">版本:</span>
              <span>{{ systemStatus.mfa?.version }}</span>
            </div>
          </el-col>

          <el-col :xs="24" :sm="12">
            <div class="label">语言模型状态:</div>
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
                  下载
                </el-button>
              </div>
            </div>
          </el-col>

          <el-col :xs="24">
            <div class="label">处理模块:</div>
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
        </el-row>
      </el-card>
    </div>

    <div v-if="systemStatus && !systemStatus.mfa?.installed" class="warning-box">
      <el-alert type="error" :closable="false" show-icon>
        <template #title>❌ MFA 未安装</template>
        <p>请先安装 Montreal Forced Aligner:</p>
        <code>pip install montreal-forced-aligner</code>
        <p style="margin-top: 10px">然后下载所需的语言模型</p>
      </el-alert>
    </div>

    <div v-if="systemStatus && systemStatus.mfa?.installed && !isReady && processingMode !== 'project-only'" class="warning-box">
      <el-alert type="warning" :closable="false" show-icon>
        <template #title>⚠️ 警告: 组件未就绪</template>
        <p>请 download 所需的语言模型或检查系统状态。</p>
      </el-alert>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, InfoFilled } from '@element-plus/icons-vue'

const emit = defineEmits(['status-changed'])

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
}

// 核心表单与模式状态（合并唯一声明）
const processingMode = ref<ProcessingMode>('mfa-only')

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
  { stage: '1. MFA 自动标注', status: '等待', message: '准备进行音频对齐' },
  { stage: '2. F0 音高提取', status: '等待', message: '提取音频基频信息' },
  { stage: '3. 工程文件生成', status: '等待', message: '生成 Synthesizer V / OpenUtau 工程文件' }
])

const currentJobId = ref<string>('')
let jobPollTimer: number | null = null

// MIDI 导入状态
const midiInfo = ref<{ bpm: number; loaded: boolean }>({ bpm: 120, loaded: false })
const midiLoaded = computed(() => processingMode.value === 'project-only' && !!formData.value.midiFile)

// 计算属性
const normalizedModels = computed(() => {
  const defaultModels = { cmn: false, eng: false, jpn: false, kor: false, yue: false }
  if (!systemStatus.value.mfa?.models || typeof systemStatus.value.mfa.models !== 'object') {
    return defaultModels
  }
  return { ...defaultModels, ...systemStatus.value.mfa.models }
})

const isReady = computed(() => {
  return !!(systemStatus.value.mfa?.installed && normalizedModels.value[formData.value.language as keyof typeof normalizedModels.value])
})

// 根据不同模式控制提交按钮的禁用状态
const isSubmitDisabled = computed(() => {
  if (processingMode.value === 'project-only') {
    return !formData.value.audioFile || !formData.value.labFile
  }
  return !formData.value.audioFile || !formData.value.text.trim() || !isReady.value
})

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
  processingDetails.value = [
    { stage: '1. MFA 自动标注', status: '等待', message: '准备进行音频对齐' },
    { stage: '2. F0 音高提取', status: '等待', message: '提取音频基频信息' },
    { stage: '3. 工程文件生成', status: '等待', message: '生成 Synthesizer V / OpenUtau 工程文件' }
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
    config: payload?.config || {
      bpm: advancedConfig.value.bpm,
      base_pitch: advancedConfig.value.base_pitch,
      auto_note_pitch: advancedConfig.value.auto_note_pitch,
      export_pitch_line: advancedConfig.value.export_pitch_line,
      f0_method: advancedConfig.value.f0_method,
      f0_device: advancedConfig.value.f0_device,
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
          throw new Error(data.error || '获取任务状态失败')
        }

        const job = data.job || {}

        if (job.status === 'done') {
          resolve(job.result || job)
          return
        } else if (job.status === 'failed') {
          throw new Error(job.error || '处理失败')
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
    const res = await fetch('/api/pipeline/status')
    const data = await res.json()
    if (data.success) {
      systemStatus.value = data.status
    }
    emit('status-changed')
  } catch (e) {
    console.warn('无法检查系统状态:', e)
    ElMessage.warning('无法连接到后端')
  } finally {
    checkingStatus.value = false
  }
}

const refreshStatus = async () => {
  await checkSystemStatus()
  ElMessage.success('已刷新系统状态')
}

const openGitHub = () => {
  window.open('https://github.com/liuhua520-svg/SVS-Lab-Aligner', '_blank')
}

const handleExceed = () => {
  ElMessage.error('只能上传一个文件')
}

const handleAudioSelect = (file: any) => {
  formData.value.audioFile = file.raw || null
}

const handleLabSelect = (file: any) => {
  formData.value.labFile = file.raw || null
}

const handleMidiSelect = (file: any) => {
  formData.value.midiFile = file.raw || null
  if (formData.value.midiFile) {
    extractMidiBpm(formData.value.midiFile).then(({ bpm }) => {
      midiInfo.value = { bpm, loaded: true }
    })
  } else {
    midiInfo.value = { bpm: 120, loaded: false }
  }
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
      ElMessage.success(`模型 ${lang} 下载成功`)
      await checkSystemStatus()
    } else {
      ElMessage.error(`模型下载失败: ${data.error}`)
    }
  } catch (e) {
    ElMessage.error(`下载模型出错: ${e}`)
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
      ElMessage.warning('请选择 WAV 文件')
      return
    }
    if (!formData.value.labFile) {
      ElMessage.warning('请选择 LAB 文件')
      return
    }

    clearJobPolling()
    processing.value = true
    progressPercent.value = 0
    error.value = ''
    result.value = null
    currentJobId.value = ''
    resetProcessingSteps()
    updateProcessingStep(0, '跳过', '工程文件模式：已跳过 MFA 自动标注')
    updateProcessingStep(1, '进行中', 'F0 提取 + 工程文件生成中，请耐心等待...')
    updateProcessingStep(2, '等待', '等待工程文件生成')

    let progressTimer: number | null = null

    try {
      const formDataObj = new FormData()
      formDataObj.append('wav_file', formData.value.audioFile)
      formDataObj.append('lab_file', formData.value.labFile)
      formDataObj.append('format', formData.value.outputFormat)
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
      // MIDI 文件（选填）
      if (formData.value.midiFile) {
        formDataObj.append('midi_file', formData.value.midiFile)
      }

      progressTimer = window.setInterval(() => {
        if (progressPercent.value < 30) progressPercent.value += 3
      }, 400)

      const res = await fetch('/api/pipeline/project-only', {
        method: 'POST',
        body: formDataObj,
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || '提交失败')

      // 异步任务轮询（后端返回 job_id）
      if (data.job_id) {
        if (progressTimer !== null) { window.clearInterval(progressTimer); progressTimer = null }
        progressPercent.value = 35

        const finalPayload = await waitForJobFinished(data.job_id)
        const normalized = normalizeResult(finalPayload)

        if (!normalized.projectPath) throw new Error('工程文件未生成，无法视为处理成功')

        result.value = normalized
        progressPercent.value = 100
        updateProcessingStep(0, '跳过', '工程文件模式不需要 MFA 标注')
        updateProcessingStep(1, '完成', 'F0 提取已完成')
        updateProcessingStep(2, '完成', `工程文件已生成: ${getFileName(normalized.projectPath)}`)
        ElMessage.success('✅ 工程文件生成成功！')
        return
      }

      // 向下兼容：同步结果回退（后端仍为同步时生效）
      if (!data.success) throw new Error(data.error || '工程文件生成失败')
      const normalized = normalizeResult(data)
      result.value = normalized
      progressPercent.value = 100
      updateProcessingStep(0, '跳过', '工程文件模式不需要 MFA 标注')
      updateProcessingStep(1, '完成', 'F0 提取已完成')
      updateProcessingStep(2, '完成', `工程文件已生成: ${getFileName(normalized.projectPath || '')}`)
      ElMessage.success('✅ 工程文件生成成功！')
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
  // 分支 2) 其他传统模式：需要输入文本和模型校验
  // ============================================================
  if (!formData.value.audioFile) {
    ElMessage.warning('请选择音频文件')
    return
  }
  if (!formData.value.text.trim()) {
    ElMessage.warning('请输入文本')
    return
  }
  if (!isReady.value) {
    ElMessage.error('系统未准备好或语言模型未下载')
    return
  }

  const maxSize = 512 * 1024 * 1024
  if (formData.value.audioFile.size > maxSize) {
    ElMessage.warning('音频文件过大（>512MB），建议分割后再处理')
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

    if (processingMode.value === 'full') {
      formDataObj.append('format', formData.value.outputFormat)
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
    }

    progressTimer = window.setInterval(() => {
      if (progressPercent.value < 30) progressPercent.value += 3
    }, 400)

    const endpoint = processingMode.value === 'full' ? '/api/pipeline/full' : '/api/pipeline/mfa-only'
    const res = await fetch(endpoint, { method: 'POST', body: formDataObj })
    const data = await res.json()

    if (!res.ok) throw new Error(data.error || '提交失败')

    // full 和 mfa-only 均走异步轮询（后端返回 job_id）
    if (data.job_id) {
      if (progressTimer !== null) { window.clearInterval(progressTimer); progressTimer = null }
      progressPercent.value = 35

      if (processingMode.value === 'mfa-only') {
        updateProcessingStep(0, '进行中', 'MFA 自动标注中，请耐心等待...')
        updateProcessingStep(1, '等待', '仅标注模式将跳过此步骤')
        updateProcessingStep(2, '等待', '仅标注模式将跳过此步骤')
      } else {
        updateProcessingStep(0, '进行中', 'MFA 标注 + F0 提取 + 工程文件生成中...')
        updateProcessingStep(1, '等待', '等待 F0 提取')
        updateProcessingStep(2, '等待', '等待工程文件生成')
      }

      const finalPayload = await waitForJobFinished(data.job_id)
      const normalized = normalizeResult(finalPayload)

      if (processingMode.value === 'full') {
        if (!normalized.projectPath) throw new Error('工程文件未生成，无法视为处理成功')
        updateProcessingStep(0, '完成', 'MFA 已完成')
        updateProcessingStep(1, '完成', 'F0 提取已完成')
        updateProcessingStep(2, '完成', `工程文件已生成: ${getFileName(normalized.projectPath)}`)
      } else {
        // mfa-only
        if (!normalized.labContent) throw new Error('LAB 内容为空，MFA 处理失败')
        const segCount = countLabSegments(normalized.labContent)
        updateProcessingStep(0, '完成', `${segCount} 个标注段`)
        updateProcessingStep(1, '跳过', '仅标注模式未执行 F0 提取')
        updateProcessingStep(2, '跳过', '仅标注模式未生成工程文件')
      }

      result.value = normalized
      progressPercent.value = 100
      ElMessage.success('✅ 处理成功！')
      return
    }

    // 向下兼容：full 模式同步结果回退
    if (processingMode.value === 'full') {
      const normalized = normalizeResult(data)
      if (data.success && normalized.projectPath) {
        updateProcessingStep(0, '完成', 'MFA 已完成')
        updateProcessingStep(1, '完成', 'F0 提取已完成')
        updateProcessingStep(2, '完成', `工程文件已生成: ${getFileName(normalized.projectPath)}`)
        result.value = normalized
        progressPercent.value = 100
        ElMessage.success('✅ 处理成功！')
        return
      }
      throw new Error(data.error || '工程文件未生成，无法视为处理成功')
    }

    // mfa-only 同步回退（后端已异步，此分支仅做兼容保留）
    if (!data.success) throw new Error(data.error || 'MFA 处理失败')
    const normalized = normalizeResult(data)
    if (!normalized.labContent) throw new Error('LAB 内容为空，MFA 处理失败')
    const segCount = countLabSegments(normalized.labContent)
    result.value = normalized
    progressPercent.value = 100
    updateProcessingStep(0, '完成', `${segCount} 个标注段`)
    updateProcessingStep(1, '跳过', '仅标注模式未执行 F0 提取')
    updateProcessingStep(2, '跳过', '仅标注模式未生成工程文件')
    ElMessage.success('✅ 处理成功！')
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
  if (!result.value?.labContent) return
  const element = document.createElement('a')
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(result.value.labContent))
  element.setAttribute('download', `alignment_${Date.now()}.lab`)
  document.body.appendChild(element)
  element.click()
  document.body.removeChild(element)
  ElMessage.success('LAB 文件已下载')
}

const downloadProject = async () => {
  if (!result.value?.projectPath) return
  downloadingProject.value = true
  try {
    const filename = result.value.projectPath.split(/[\\/]/).pop()
    const response = await fetch(`/api/work-dir/download/${encodeURIComponent(filename)}`)
    if (!response.ok) {
      ElMessage.error('下载失败')
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
    ElMessage.success('工程文件已下载')
  } catch (e) {
    ElMessage.error(`下载失败: ${e}`)
  } finally {
    downloadingProject.value = false
  }
}

const copyLabToClipboard = () => {
  if (!result.value?.labContent) return
  navigator.clipboard.writeText(result.value.labContent).then(() => {
    ElMessage.success('已复制到剪贴板')
  }).catch(() => {
    ElMessage.error('复制失败')
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
    projectTitle: 'Project',
    phonemeMode: 'none'
  }
  midiInfo.value = { bpm: 120, loaded: false }
  result.value = null
  error.value = ''
  progressPercent.value = 0
  currentJobId.value = ''
  resetProcessingSteps()
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

.processor-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
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

/* 确保模式帮助文本在 Element 表单条目中强制换行，不向右侧外溢 */
.mode-help {
  width: 100%;
  display: block;
  color: #909399;
  font-size: 12px;
  margin-top: 6px;
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
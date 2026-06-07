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

        <el-form-item label="输入文本">
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

        <el-form-item label="语言">
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
          </el-radio-group>
          <div class="mode-help">
            <small v-if="processingMode === 'mfa-only'">
              只进行 MFA 自动标注，生成 LAB 文件
            </small>
            <small v-else>
              执行完整流程：标注 → F0提取 → 工程文件生成
            </small>
          </div>
        </el-form-item>

        <el-form-item v-if="processingMode === 'full'" label="输出格式">
          <el-select v-model="formData.outputFormat" placeholder="选择输出格式">
            <el-option label="Synthesizer V Studio (.svp)" value="sv" />
            <el-option label="OpenUtau/UTAU (.ustx)" value="utau" />
          </el-select>
        </el-form-item>

        <el-form-item v-if="processingMode === 'full'" label="工程标题">
          <el-input
            v-model="formData.projectTitle"
            placeholder="输入工程文件标题"
            maxlength="248"
          />
        </el-form-item>

        <el-collapse v-if="processingMode === 'full'" accordion>
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
                  />
                </el-form-item>
              </el-col>

              <el-col :xs="24" :sm="12">
                <el-form-item label="基准音高 (MIDI Note)">
                  <div class="pitch-input-group">
                    <el-input-number
                      v-model="advancedConfig.base_pitch"
                      :min="21"
                      :max="108"
                      :step="1"
                      controls-position="right"
                    />
                    <span class="pitch-name">{{ midiNoteToName(advancedConfig.base_pitch) }}</span>
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
                  />
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
              <p><strong>DIO vs Harvest:</strong> DIO 快速但可能不够精确；Harvest 慢但更精确</p>
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
            :disabled="!formData.audioFile || !formData.text || !isReady"
          >
            <span v-if="!processing">🚀 开始处理</span>
            <span v-else>处理中... {{ progressPercent }}%</span>
          </el-button>
          <el-button @click="reset" :disabled="processing">🔄 重置</el-button>
          <span v-if="!isReady" class="disabled-text">
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
          <el-tab-pane label="LAB 标注内容">
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
                <el-col :xs="24">
                  <p><strong>LAB 标注文件:</strong></p>
                  <code>{{ result.labPath }}</code>
                </el-col>
                <el-col :xs="24">
                  <p><strong>工程文件:</strong></p>
                  <code>{{ result.projectPath }}</code>
                </el-col>
                <el-col :xs="24">
                  <p><strong>输出格式:</strong> {{ result.projectFormat === 'sv' ? 'Synthesizer V Studio' : 'OpenUtau/UTAU' }}</p>
                  <p><strong>标注段数:</strong> {{ result.segments }}</p>
                  <p v-if="result.config"><strong>处理配置:</strong></p>
                  <ul v-if="result.config">
                    <li>BPM: {{ result.config.bpm }}</li>
                    <li>基准音高: {{ midiNoteToName(result.config.base_pitch) }} (MIDI {{ result.config.base_pitch }})</li>
                    <li>自动音符音高: {{ result.config.auto_note_pitch ? '已启用' : '已禁用' }}</li>
                    <li>导出连续音高: {{ result.config.export_pitch_line ? '已启用' : '已禁用' }}</li>
                    <li>F0 方法: {{ result.config.f0_method }}</li>
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
                    <el-tag v-else type="info">{{ row.status }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="message" label="详情" show-overflow-tooltip />
              </el-table>
            </div>
          </el-tab-pane>
        </el-tabs>

        <div class="action-buttons">
          <el-button type="success" @click="downloadLab" size="large">
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
          <el-button @click="copyLabToClipboard" size="large">
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
                  PyWORLD (F0提取): {{ systemStatus.audio_processing?.pyworld_available ? '✓' : '✗' }}
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

    <div v-if="systemStatus && systemStatus.mfa?.installed && !isReady" class="warning-box">
      <el-alert type="warning" :closable="false" show-icon>
        <template #title>⚠️ 警告: 组件未就绪</template>
        <p>请下载所需的语言模型或检查系统状态。</p>
      </el-alert>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled, InfoFilled } from '@element-plus/icons-vue'

const emit = defineEmits(['status-changed'])

interface FormData {
  audioFile: File | null
  text: string
  language: string
  outputFormat: string
  projectTitle: string
}

interface AdvancedConfig {
  bpm: number
  base_pitch: number
  auto_note_pitch: boolean    // 新增：自动音符音高控制
  export_pitch_line: boolean  // 新增：导出连续音高曲线控制
  f0_method: 'dio' | 'harvest'
  precision: 'single' | 'double'
  f0_smooth: boolean
  f0_smooth_window: number
  f0_floor: number
  f0_ceil: number
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
  }
}

const formData = ref<FormData>({
  audioFile: null,
  text: '',
  language: 'cmn',
  outputFormat: 'sv',
  projectTitle: 'Project'
})

const advancedConfig = ref<AdvancedConfig>({
  bpm: 120,
  base_pitch: 60,
  auto_note_pitch: true,    // 默认开启自动对齐实际音高
  export_pitch_line: true,  // 默认开启导出 F0 曲线参数
  f0_method: 'harvest',
  precision: 'double',
  f0_smooth: true,
  f0_smooth_window: 5,
  f0_floor: 71,
  f0_ceil: 800
})

const processingMode = ref('mfa-only')
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
    models: {
      cmn: false,
      eng: false,
      jpn: false,
      kor: false,
      yue: false
    }
  },
  audio_processing: {
    pyworld_available: false,
    supported_formats: []
  }
})

const processingDetails = ref<any[]>([
  { stage: '1. MFA 自动标注', status: '等待', message: '準备進行音頻對齐' },
  { stage: '2. F0 音高提取', status: '等待', message: '提取音频基频信息' },
  { stage: '3. 工程文件生成', status: '等待', message: '生成 Synthesizer V / OpenUtau 工程文件' }
])

// 防御性计算属性
const normalizedModels = computed(() => {
  const defaultModels = {
    cmn: false,
    eng: false,
    jpn: false,
    kor: false,
    yue: false
  }
  
  if (!systemStatus.value.mfa?.models || typeof systemStatus.value.mfa.models !== 'object') {
    return defaultModels
  }
  
  return { ...defaultModels, ...systemStatus.value.mfa.models }
})

const isReady = computed(() => {
  return systemStatus.value.mfa?.installed && 
         normalizedModels.value[formData.value.language as keyof typeof normalizedModels.value]
})

onMounted(() => {
  checkSystemStatus()
})

// MIDI 音符转换为音名
const midiNoteToName = (note: number): string => {
  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  const octave = Math.floor(note / 12) - 1
  const noteName = notes[note % 12]
  return `${noteName}${octave}`
}

// 格式化文件大小
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
}

// 格式化时间
const formatTime = (ms: number): string => {
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`
  } else {
    return `${seconds}s`
  }
}

// 获取文件名
const getFileName = (path: string): string => {
  return path.split(/[\\/]/).pop() || path
}

const checkSystemStatus = async () => {
  checkingStatus.value = true
  try {
    const res = await fetch('/api/pipeline/status')
    const data = await res.json()
    
    if (data.success) {
      systemStatus.value = data.status
    } else {
      console.warn('获取状态失败:', data.error)
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
  window.open('https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner', '_blank')
}

const handleExceed = (files: File[]) => {
  ElMessage.error('只能上传一个文件')
}

const downloadModel = async (lang: string) => {
  downloadingLangs.value.push(lang)
  try {
    const res = await fetch(`/api/mfa/download-model/${lang}`, {
      method: 'POST'
    })
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

const handleAudioSelect = (file: any) => {
  formData.value.audioFile = file.raw
}

const processAudio = async () => {
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

  // 检查音频文件大小
  const maxSize = 512 * 1024 * 1024 // 512MB
  if (formData.value.audioFile.size > maxSize) {
    ElMessage.warning('音频文件过大（>512MB），建议分割后再处理')
  }

  processing.value = true
  progressPercent.value = 0
  error.value = ''
  processingDetails.value = processingDetails.value.map(d => ({
    ...d,
    status: '等待'
  }))

  try {
    const formDataObj = new FormData()
    formDataObj.append('audio_file', formData.value.audioFile)
    formDataObj.append('text', formData.value.text)
    formDataObj.append('language', formData.value.language)

    if (processingMode.value === 'full') {
      formDataObj.append('format', formData.value.outputFormat)
      formDataObj.append('title', formData.value.projectTitle)
      
      // 传递包含细化音高的全套高级配置到后端
      formDataObj.append('bpm', advancedConfig.value.bpm.toString())
      formDataObj.append('base_pitch', advancedConfig.value.base_pitch.toString())
      formDataObj.append('auto_note_pitch', advancedConfig.value.auto_note_pitch.toString())
      formDataObj.append('export_pitch_line', advancedConfig.value.export_pitch_line.toString())
      formDataObj.append('f0_method', advancedConfig.value.f0_method)
      formDataObj.append('precision', advancedConfig.value.precision)
      formDataObj.append('f0_smooth', advancedConfig.value.f0_smooth.toString())
      formDataObj.append('f0_smooth_window', advancedConfig.value.f0_smooth_window.toString())
    }

    // 模拟进度
    const progressInterval = setInterval(() => {
      if (progressPercent.value < 95) {
        progressPercent.value += Math.random() * 10
      }
    }, 500)

    const endpoint = processingMode.value === 'full' 
      ? '/api/pipeline/full'
      : '/api/pipeline/mfa-only'

    const res = await fetch(endpoint, {
      method: 'POST',
      body: formDataObj
    })

    clearInterval(progressInterval)
    progressPercent.value = 100

    const data = await res.json()

    if (data.success) {
      result.value = {
        labContent: data.lab_content || '',
        processingTime: data.processing_time || data.processing_time_ms || 0,
        labPath: data.lab_path,
        projectPath: data.project_path,
        projectFormat: data.project_format,
        segments: data.segments,
        config: data.config || { // 兜底防止后端未回传
          bpm: advancedConfig.value.bpm,
          base_pitch: advancedConfig.value.base_pitch,
          auto_note_pitch: advancedConfig.value.auto_note_pitch,
          export_pitch_line: advancedConfig.value.export_pitch_line,
          f0_method: advancedConfig.value.f0_method,
          use_double_precision: advancedConfig.value.precision === 'double'
        }
      }
      
      // 根据勾选细化逻辑动态更新显示步骤状态
      const isF0StepActive = processingMode.value === 'full' && (advancedConfig.value.export_pitch_line || advancedConfig.value.auto_note_pitch)
      
      processingDetails.value = [
        { stage: '1. MFA 自动标注', status: '完成', message: `${data.segments || '?'} 个标注段` },
        { stage: '2. F0 音高提取', status: isF0StepActive ? '完成' : '跳过', message: isF0StepActive ? '基频信息已处理' : '配置已选择跳过提取' },
        { stage: '3. 工程文件生成', status: processingMode.value === 'full' ? '完成' : '跳过', message: data.project_format ? `${data.project_format.toUpperCase()} 格式` : '-' }
      ]

      ElMessage.success('✅ 处理成功！')
    } else {
      error.value = data.error || '处理失败'
      ElMessage.error(`❌ ${error.value}`)
    }
  } catch (e) {
    error.value = String(e)
    ElMessage.error(`❌ 错误: ${e}`)
  } finally {
    processing.value = false
  }
}

const downloadLab = () => {
  if (!result.value?.labContent) return

  const element = document.createElement('a')
  element.setAttribute(
    'href',
    'data:text/plain;charset=utf-8,' + encodeURIComponent(result.value.labContent)
  )
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
  formData.value = {
    audioFile: null,
    text: '',
    language: 'cmn',
    outputFormat: 'sv',
    projectTitle: 'Project'
  }
  result.value = null
  error.value = ''
  processingDetails.value = processingDetails.value.map(d => ({
    ...d,
    status: '等待'
  }))
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

.mode-help {
  color: #909399;
  font-size: 12px;
  margin-top: 8px;
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
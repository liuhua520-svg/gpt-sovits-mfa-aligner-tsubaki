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
            <el-tooltip content="检查MFA状态" placement="bottom">
              <el-button link @click="refreshMFAStatus" :loading="checkingStatus">
                🔄 检查状态
              </el-button>
            </el-tooltip>
          </div>
        </div>
      </template>

      <el-form :model="formData" label-width="100px">
        <!-- 音频上传 -->
        <el-form-item label="音频文件">
          <el-upload
            drag
            action="#"
            :auto-upload="false"
            :limit="1"
            :on-exceed="handleExceed"
            @change="handleAudioSelect"
            accept=".wav,.mp3,.flac"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              拖拽或<em>点击选择</em>音频文件
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持 WAV / MP3 / FLAC
              </div>
            </template>
          </el-upload>
          <div v-if="formData.audioFile" class="file-info">
            ✓ {{ formData.audioFile.name }}
          </div>
        </el-form-item>

        <!-- 文本输入 -->
        <el-form-item label="输入文本">
          <el-input
            v-model="formData.text"
            type="textarea"
            :rows="4"
            placeholder="粘贴GPT-SOVITS的输入文本"
            show-word-limit
          />
          <div class="help-text">
            💡 将GPT-SOVITS中用于生成语音的文本复制粘贴到这里
          </div>
        </el-form-item>

        <!-- 语言选择 -->
        <el-form-item label="语言">
          <el-select v-model="formData.language">
            <el-option label="普通话 🇨🇳" value="cmn" />
            <el-option label="英语 🇬🇧" value="eng" />
            <el-option label="日语 🇯🇵" value="jpn" />
            <el-option label="韩语 🇰🇷" value="kor" />
            <el-option label="粤语 🇭🇰" value="yue" />
          </el-select>
        </el-form-item>

        <!-- 处理按钮 -->
        <el-form-item>
          <el-button
            type="primary"
            size="large"
            :loading="processing"
            @click="processAudio"
            :disabled="!formData.audioFile || !formData.text || !mfaReady"
          >
            <span v-if="!processing">🚀 开始标注</span>
            <span v-else>处理中... {{ progressPercent }}%</span>
          </el-button>
          <el-button @click="reset" :disabled="processing">重置</el-button>
        </el-form-item>

        <!-- 进度条 -->
        <el-progress
          v-if="processing"
          :percentage="progressPercent"
          :indeterminate="true"
        />
      </el-form>

      <!-- 结果显示 -->
      <div v-if="result" class="result-section">
        <el-divider />

        <h3>✅ 标注结果</h3>
        <div class="result-info">
          <p>处理时间: {{ result.processingTime }}ms</p>
        </div>

        <el-input
          v-model="result.labContent"
          type="textarea"
          :rows="10"
          readonly
          class="lab-output"
        />

        <!-- 操作按钮 -->
        <div class="action-buttons">
          <el-button type="success" @click="downloadLab" size="large">
            📥 下载LAB文件
          </el-button>
          <el-button @click="copyToClipboard" size="large">
            📋 复制内容
          </el-button>
          <el-button type="info" @click="newProcess" size="large">
            🔄 处理下一个
          </el-button>
        </div>
      </div>

      <!-- 错误提示 -->
      <div v-if="error" class="error-section">
        <el-alert
          :title="`错误: ${error}`"
          type="error"
          :closable="true"
          @close="error = ''"
        />
      </div>
    </el-card>

    <!-- MFA状态提示 -->
    <div v-if="mfaStatus" class="status-box">
      <el-card shadow="hover">
        <template #header>
          <span>🔧 MFA 状态</span>
        </template>

        <el-row :gutter="20">
          <el-col :xs="24" :sm="12">
            <div class="status-item">
              <span class="label">MFA 安装状态:</span>
              <el-tag :type="mfaStatus.installed ? 'success' : 'danger'" size="large">
                {{ mfaStatus.installed ? '✓ 已安装' : '✗ 未安装' }}
              </el-tag>
            </div>
            <div v-if="mfaStatus.installed" class="status-item">
              <span class="label">版本:</span>
              <span>{{ mfaStatus.version }}</span>
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
        </el-row>
      </el-card>
    </div>

    <!-- MFA未安装提示 -->
    <div v-if="!mfaStatus.installed" class="warning-box">
      <el-alert type="error" :closable="false" show-icon>
        <template #title>❌ MFA 未安装</template>
        <p>请先安装 Montreal Forced Aligner:</p>
        <code>pip install montreal-forced-aligner</code>
        <p style="margin-top: 10px">然后下载所需的语言模型:</p>
        <code>mfa model download acoustic cmn  # 中文</code>
        <code>mfa model download acoustic eng  # 英语</code>
        <code>mfa model download acoustic jpn  # 日语</code>
      </el-alert>
    </div>

    <!-- 其他必需模块未安装提示 -->
    <div v-if="mfaStatus.installed && !mfaReady" class="warning-box">
      <el-alert type="warning" :closable="false" show-icon>
        <template #title>⚠️ 警告: 所选语言模型未下载</template>
        <p>请先下载所需的语言模型或检查MFA状态。</p>
      </el-alert>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

const emit = defineEmits(['mfa-status-changed'])

interface FormData {
  audioFile: File | null
  text: string
  language: string
}

interface MFAStatus {
  installed: boolean
  version: string
  models?: {
    cmn?: boolean
    eng?: boolean
    jpn?: boolean
    kor?: boolean
    yue?: boolean
  }
}

const formData = ref<FormData>({
  audioFile: null,
  text: '',
  language: 'cmn'
})

const processing = ref(false)
const progressPercent = ref(0)
const result = ref<any>(null)
const error = ref('')
const checkingStatus = ref(false)
const downloadingLangs = ref<string[]>([])
const mfaStatus = ref<MFAStatus>({
  installed: false,
  version: 'unknown',
  models: {
    cmn: false,
    eng: false,
    jpn: false,
    kor: false,
    yue: false
  }
})

// 防御性计算属性：确保 models 存在且有正确结构
const normalizedModels = computed(() => {
  const defaultModels = {
    cmn: false,
    eng: false,
    jpn: false,
    kor: false,
    yue: false
  }
  
  // 如果 models 不存在或不是对象，返回全部 false
  if (!mfaStatus.value.models || typeof mfaStatus.value.models !== 'object') {
    return defaultModels
  }
  
  // 合并现有模型状态和默认值
  return { ...defaultModels, ...mfaStatus.value.models }
})

const mfaReady = computed(() => {
  return mfaStatus.value.installed && 
         normalizedModels.value[formData.value.language as keyof typeof normalizedModels.value]
})

onMounted(() => {
  checkMFAStatus()
})

const checkMFAStatus = async () => {
  checkingStatus.value = true
  try {
    const res = await fetch('/api/mfa/status')
    const data = await res.json()
    
    // 确保响应有必要的字段
    if (!data.models) {
      data.models = {
        cmn: false,
        eng: false,
        jpn: false,
        kor: false,
        yue: false
      }
    }
    
    mfaStatus.value = data
    emit('mfa-status-changed')
  } catch (e) {
    console.warn('无法检查MFA状态:', e)
    ElMessage.warning('无法连接到后端，请检查服务是否运行')
  } finally {
    checkingStatus.value = false
  }
}

const refreshMFAStatus = async () => {
  await checkMFAStatus()
  ElMessage.success('已刷新MFA状态')
}

const openGitHub = () => {
  window.open('https://github.com/xiaofan310-vb/gpt-sovits-mfa-aligner', '_blank')
}

const handleExceed = (files: File[]) => {
  ElMessage.error('只能上传一个文件，请移除其他文件后重试')
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
      await checkMFAStatus()
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

  if (!mfaReady.value) {
    ElMessage.error('MFA未准备好或语言模型未下载')
    return
  }

  processing.value = true
  progressPercent.value = 0
  error.value = ''

  try {
    const formDataObj = new FormData()
    formDataObj.append('audio_file', formData.value.audioFile)
    formDataObj.append('text', formData.value.text)
    formDataObj.append('language', formData.value.language)

    // 模拟进度
    const progressInterval = setInterval(() => {
      if (progressPercent.value < 95) {
        progressPercent.value += Math.random() * 15
      }
    }, 300)

    const res = await fetch('/api/mfa/process', {
      method: 'POST',
      body: formDataObj
    })

    clearInterval(progressInterval)
    progressPercent.value = 100

    const data = await res.json()

    if (data.success) {
      result.value = {
        labContent: data.lab_content,
        processingTime: data.processing_time_ms
      }
      ElMessage.success('✅ 标注成功！')
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
  if (!result.value) return

  const element = document.createElement('a')
  element.setAttribute(
    'href',
    'data:text/plain;charset=utf-8,' + encodeURIComponent(result.value.labContent)
  )
  element.setAttribute('download', 'alignment.lab')
  document.body.appendChild(element)
  element.click()
  document.body.removeChild(element)
  ElMessage.success('LAB文件已下载')
}

const copyToClipboard = () => {
  if (!result.value) return

  navigator.clipboard.writeText(result.value.labContent).then(() => {
    ElMessage.success('已复制到剪贴板')
  })
}

const reset = () => {
  formData.value = {
    audioFile: null,
    text: '',
    language: 'cmn'
  }
  result.value = null
  error.value = ''
}

const newProcess = () => {
  reset()
}
</script>

<style scoped>
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

.result-section {
  margin-top: 30px;
  padding-top: 20px;
}

.result-info {
  margin-bottom: 15px;
  padding: 10px;
  background: #f0f9ff;
  border-radius: 4px;
}

.result-info p {
  margin: 0;
  color: #606266;
  font-size: 12px;
}

.lab-output {
  font-family: 'Courier New', monospace;
  font-size: 11px;
  margin: 15px 0;
}

.action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  flex-wrap: wrap;
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
}
</style>

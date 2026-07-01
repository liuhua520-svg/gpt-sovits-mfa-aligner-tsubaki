<template>
  <div class="dialogue-batch-container">
    <el-card class="batch-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span class="card-title">💬 {{ t('dialogue.pageTitle') }}</span>
          <div class="header-actions">
            <el-button @click="importFolder" type="primary" link>📁 {{ t('dialogue.importFolder') }}</el-button>
            <el-button @click="clearAll" type="danger" link>🗑️ {{ t('dialogue.clearAll') }}</el-button>
          </div>
        </div>
      </template>

      <el-alert type="info" :closable="false" show-icon class="subtitle">
        <template #title>{{ t('dialogue.pageSubtitle') }}</template>
      </el-alert>

      <!-- 高级设置（与单文件处理相同） -->
      <el-collapse accordion style="margin-bottom: 20px">
        <el-collapse-item :title="`⚙️ ${t('processor.advancedSettingsTitle')}`" name="advanced">
          <el-row :gutter="20">
            <el-col :xs="24" :sm="12">
              <el-form-item :label="t('processor.bpm')">
                <el-input-number v-model="batchConfig.bpm" :min="20" :max="300" :step="1" />
              </el-form-item>
            </el-col>
            <el-col :xs="24" :sm="12">
              <el-form-item :label="t('processor.basePitch')">
                <el-input-number v-model="batchConfig.basePitch" :min="12" :max="108" :step="1" />
              </el-form-item>
            </el-col>
            <el-col :xs="24" :sm="12">
              <el-form-item :label="t('processor.f0Method')">
                <el-select v-model="batchConfig.f0Method">
                  <el-option value="dio" label="DIO" />
                  <el-option value="harvest" label="Harvest" />
                  <el-option value="crepe" label="CREPE" />
                  <el-option value="rmvpe" label="RMVPE" />
                </el-select>
              </el-form-item>
            </el-col>
            <el-col :xs="24" :sm="12">
              <el-form-item :label="t('processor.outputFormat')">
                <el-select v-model="batchConfig.outputFormat">
                  <el-option value="sv" :label="t('processor.outputFormatSv')" />
                  <el-option value="utau" :label="t('processor.outputFormatUtau')" />
                  <el-option value="vsqx" :label="t('processor.outputFormatVsqx')" />
                </el-select>
              </el-form-item>
            </el-col>
          </el-row>
        </el-collapse-item>
      </el-collapse>

      <!-- 对话框列表 -->
      <div class="dialogue-list">
        <div v-for="(box, idx) in dialogueBoxes" :key="idx" class="dialogue-box">
          <div class="box-header">
            <span class="box-index">{{ t('dialogue.boxIndex', { index: idx + 1 }) }}</span>
            <el-tag :type="getBoxStatusType(box.status)" size="small">
              {{ getBoxStatusLabel(box.status) }}
            </el-tag>
            <el-button 
              v-if="dialogueBoxes.length > 1" 
              link 
              type="danger" 
              size="small"
              @click="removeBox(idx)"
            >
              ✕
            </el-button>
          </div>

          <el-row :gutter="10" class="box-content">
            <el-col :xs="24" :sm="12">
              <el-input 
                v-model="box.text" 
                type="textarea" 
                :rows="4"
                :placeholder="t('dialogue.textPlaceholder')"
                :disabled="processing"
              />
            </el-col>
            <el-col :xs="24" :sm="12">
              <div class="upload-section">
                <div class="upload-label">{{ t('dialogue.audioLabel') }}</div>
                <el-upload
                  :auto-upload="false"
                  :limit="1"
                  accept=".wav,.mp3,.flac,.m4a,.aac,.ogg"
                  @change="(file) => handleAudioSelect(idx, file)"
                  drag
                >
                  <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                  <div class="el-upload__text">{{ t('dialogue.dragAudio') }}</div>
                </el-upload>
                <div v-if="box.audioFile" class="file-info">✓ {{ box.audioFile.name }}</div>
              </div>

              <div class="upload-section">
                <div class="upload-label">{{ t('dialogue.labLabel') }}</div>
                <el-upload
                  :auto-upload="false"
                  :limit="1"
                  accept=".lab"
                  @change="(file) => handleLabSelect(idx, file)"
                  drag
                >
                  <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                  <div class="el-upload__text">{{ t('dialogue.dragLab') }}</div>
                </el-upload>
                <div v-if="box.labFile" class="file-info">📄 {{ box.labFile.name }}</div>
              </div>
            </el-col>
          </el-row>

          <div v-if="box.error" class="box-error">❌ {{ box.error }}</div>
        </div>
      </div>

      <!-- 添加新对话框按钮 -->
      <el-button 
        v-if="dialogueBoxes.length < 64" 
        @click="addBox" 
        style="width: 100%; margin-top: 15px"
      >
        + {{ t('dialogue.addBox') }}
      </el-button>

      <!-- 处理按钮 -->
      <div class="action-buttons">
        <el-button 
          type="primary" 
          size="large" 
          @click="startProcessing"
          :loading="processing"
          :disabled="!canStartProcessing"
        >
          🚀 {{ processing ? t('dialogue.stopProcessing') : t('dialogue.startProcessing') }}
        </el-button>
      </div>

      <!-- 进度显示 -->
      <div v-if="processing" class="progress-section">
        <el-progress 
          :percentage="Math.round((processedCount / dialogueBoxes.length) * 100)" 
          :indeterminate="true"
        />
        <div class="progress-text">
          {{ t('dialogue.processingProgress', { done: processedCount, total: dialogueBoxes.length }) }}
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useI18n } from 'vue-i18n'
import { UploadFilled } from '@element-plus/icons-vue'

interface DialogueBox {
  text: string
  audioFile: File | null
  labFile: File | null
  status: 'idle' | 'queued' | 'processing' | 'done' | 'failed'
  error: string | null
  jobId: string | null
}

interface BatchConfig {
  bpm: number
  basePitch: number
  f0Method: string
  outputFormat: string
}

const { t } = useI18n()

const dialogueBoxes = ref<DialogueBox[]>(
  Array.from({ length: 32 }, () => ({
    text: '',
    audioFile: null,
    labFile: null,
    status: 'idle' as const,
    error: null,
    jobId: null
  }))
)

const batchConfig = ref<BatchConfig>({
  bpm: 120,
  basePitch: 60,
  f0Method: 'dio',
  outputFormat: 'sv'
})

const processing = ref(false)
const processedCount = ref(0)

const canStartProcessing = computed(() => {
  return !processing.value && dialogueBoxes.value.some(box => box.audioFile)
})

const addBox = () => {
  if (dialogueBoxes.value.length < 64) {
    dialogueBoxes.value.push({
      text: '',
      audioFile: null,
      labFile: null,
      status: 'idle',
      error: null,
      jobId: null
    })
  }
}

const removeBox = (idx: number) => {
  if (dialogueBoxes.value.length > 1) {
    dialogueBoxes.value.splice(idx, 1)
  }
}

const handleAudioSelect = (idx: number, file: any) => {
  dialogueBoxes.value[idx].audioFile = file.raw
}

const handleLabSelect = (idx: number, file: any) => {
  dialogueBoxes.value[idx].labFile = file.raw
}

const importFolder = () => {
  ElMessage.info('文件夹导入功能开发中...')
}

const clearAll = async () => {
  try {
    await ElMessageBox.confirm(
      t('dialogue.confirmClearAll'),
      t('dialogue.clearAll'),
      { type: 'warning' }
    )
    dialogueBoxes.value = Array.from({ length: 32 }, () => ({
      text: '',
      audioFile: null,
      labFile: null,
      status: 'idle',
      error: null,
      jobId: null
    }))
  } catch {
    // 用户取消
  }
}

const getBoxStatusType = (status: string) => {
  const types: Record<string, string> = {
    idle: 'info',
    queued: 'warning',
    processing: 'warning',
    done: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

const getBoxStatusLabel = (status: string) => {
  const labels: Record<string, string> = {
    idle: t('dialogue.boxStatusIdle'),
    queued: t('dialogue.boxStatusQueued'),
    processing: t('dialogue.boxStatusProcessing'),
    done: t('dialogue.boxStatusDone'),
    failed: t('dialogue.boxStatusFailed')
  }
  return labels[status] || status
}

const startProcessing = async () => {
  const hasAudio = dialogueBoxes.value.some(box => box.audioFile)
  if (!hasAudio) {
    ElMessage.warning(t('dialogue.emptyBoxesWarning'))
    return
  }

  processing.value = true
  processedCount.value = 0

  try {
    // 顺序处理每个对话框
    for (let idx = 0; idx < dialogueBoxes.value.length; idx++) {
      const box = dialogueBoxes.value[idx]
      if (!box.audioFile) {
        box.status = 'idle'
        continue
      }

      box.status = 'queued'
      box.error = null

      try {
        // 发送请求
        const formData = new FormData()
        formData.append('audio_file', box.audioFile)
        formData.append('text', box.text)
        formData.append('language', 'cmn')
        formData.append('format', batchConfig.value.outputFormat)
        formData.append('title', `Dialogue_${idx + 1}`)
        formData.append('bpm', batchConfig.value.bpm.toString())
        formData.append('base_pitch', batchConfig.value.basePitch.toString())
        formData.append('f0_method', batchConfig.value.f0Method)
        // ... 其他参数

        const res = await fetch('/api/pipeline/full', {
          method: 'POST',
          body: formData
        })

        const data = await res.json()
        if (data.job_id) {
          box.jobId = data.job_id
          box.status = 'processing'
          await waitForJobFinished(idx, data.job_id)
        } else {
          throw new Error(data.error || '提交失败')
        }
      } catch (e) {
        box.status = 'failed'
        box.error = String(e)
      }

      processedCount.value++
    }

    ElMessage.success('批处理完成！')
  } finally {
    processing.value = false
  }
}

const waitForJobFinished = async (boxIdx: number, jobId: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/pipeline/job/${jobId}`)
        const data = await res.json()

        if (!data.success) {
          throw new Error(data.error || '获取状态失败')
        }

        const job = data.job || {}
        if (job.status === 'done') {
          dialogueBoxes.value[boxIdx].status = 'done'
          resolve()
        } else if (job.status === 'failed') {
          throw new Error(job.error || '处理失败')
        } else {
          // 继续轮询
          setTimeout(poll, 1500)
        }
      } catch (e) {
        dialogueBoxes.value[boxIdx].status = 'failed'
        dialogueBoxes.value[boxIdx].error = String(e)
        reject(e)
      }
    }
    poll()
  })
}

onMounted(() => {
  // 初始化
})
</script>

<style scoped>
.dialogue-batch-container {
  width: 100%;
  padding: 20px;
}

.batch-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.card-title {
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

.header-actions {
  display: flex;
  gap: 15px;
}

.subtitle {
  margin-bottom: 20px;
}

.dialogue-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
  margin: 20px 0;
  max-height: 600px;
  overflow-y: auto;
}

.dialogue-box {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 15px;
  background: #fafafa;
}

.box-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.box-index {
  font-weight: bold;
  color: #333;
  font-size: 14px;
}

.box-content {
  margin-bottom: 10px;
}

.upload-section {
  margin-bottom: 15px;
}

.upload-label {
  color: #606266;
  font-size: 12px;
  margin-bottom: 8px;
  font-weight: bold;
}

.file-info {
  color: #67c23a;
  font-size: 12px;
  margin-top: 6px;
}

.box-error {
  color: #f56c6c;
  font-size: 12px;
  margin-top: 10px;
}

.action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
}

.action-buttons :deep(.el-button) {
  flex: 1;
  min-width: 200px;
}

.progress-section {
  margin-top: 20px;
  padding: 15px;
  background: #f0f9ff;
  border-radius: 4px;
}

.progress-text {
  text-align: center;
  color: #606266;
  font-size: 12px;
  margin-top: 10px;
}
</style>
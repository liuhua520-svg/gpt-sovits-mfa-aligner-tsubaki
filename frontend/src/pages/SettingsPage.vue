<template>
  <div class="settings-page-container">
    <el-card class="settings-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span class="card-title">⚙️ {{ t('settings.pageTitle') }}</span>
        </div>
      </template>

      <el-alert type="info" :closable="false" show-icon class="subtitle">
        <template #title>{{ t('settings.pageSubtitle') }}</template>
      </el-alert>

      <el-form :model="settings" label-width="200px" class="settings-form">
        <!-- 模型自动更新 -->
        <el-form-item :label="t('settings.autoUpdateModels')">
          <el-switch 
            v-model="settings.autoUpdateModels"
            :active-text="t('processor.enabled')"
            :inactive-text="t('processor.disabled')"
          />
          <div class="hint">{{ t('settings.autoUpdateModelsHint') }}</div>
        </el-form-item>

        <el-divider />

        <!-- 镜像站配置 -->
        <el-form-item :label="t('settings.useMirror')">
          <el-switch 
            v-model="settings.useMirror"
            :active-text="t('processor.enabled')"
            :inactive-text="t('processor.disabled')"
          />
          <div class="hint">{{ t('settings.useMirrorHint') }}</div>
        </el-form-item>

        <el-form-item v-if="settings.useMirror" :label="t('settings.mirrorUrl')">
          <el-input 
            v-model="settings.mirrorUrl"
            :placeholder="t('settings.mirrorUrlPlaceholder')"
            clearable
          />
        </el-form-item>

        <el-divider />

        <!-- 保存按钮 -->
        <el-form-item>
          <el-button type="primary" size="large" @click="saveSettings" :loading="saving">
            💾 {{ t('settings.saveButton') }}
          </el-button>
        </el-form-item>

        <!-- 重启提示 -->
        <el-alert type="warning" :closable="false" show-icon class="restart-hint">
          <template #title>{{ t('settings.restartHint') }}</template>
        </el-alert>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'

interface SettingsData {
  autoUpdateModels: boolean
  useMirror: boolean
  mirrorUrl: string
}

const { t } = useI18n()
const saving = ref(false)
const settings = ref<SettingsData>({
  autoUpdateModels: true,
  useMirror: false,
  mirrorUrl: 'https://hf-mirror.com/'
})

const loadSettings = async () => {
  try {
    const res = await fetch('/api/settings')
    const data = await res.json()
    if (data.success && data.settings) {
      settings.value = {
        autoUpdateModels: data.settings.autoUpdateModels !== false,
        useMirror: data.settings.useMirror === true,
        mirrorUrl: data.settings.mirrorUrl || 'https://hf-mirror.com/'
      }
    }
  } catch (e) {
    ElMessage.error(t('settings.loadFailed', { error: String(e) }))
  }
}

const saveSettings = async () => {
  saving.value = true
  try {
    const payload = {
      autoUpdateModels: settings.value.autoUpdateModels,
      useMirror: settings.value.useMirror,
      mirrorUrl: settings.value.useMirror ? settings.value.mirrorUrl : ''
    }

    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    const data = await res.json()
    if (data.success) {
      ElMessage.success(t('settings.saveSuccess'))
    } else {
      ElMessage.error(data.error || t('settings.saveFailed', { error: '未知错误' }))
    }
  } catch (e) {
    ElMessage.error(t('settings.saveFailed', { error: String(e) }))
  } finally {
    saving.value = false
  }
}

onMounted(() => {
  loadSettings()
})
</script>

<style scoped>
.settings-page-container {
  width: 100%;
  padding: 20px;
}

.settings-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  max-width: 800px;
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

.subtitle {
  margin-bottom: 20px;
}

.settings-form {
  margin-top: 20px;
}

.hint {
  color: #909399;
  font-size: 12px;
  margin-top: 6px;
  line-height: 1.5;
}

.restart-hint {
  margin-top: 20px;
}
</style>
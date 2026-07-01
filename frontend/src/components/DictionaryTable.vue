<template>
  <div class="dictionary-table-container">
    <div class="toolbar">
      <el-button type="primary" @click="showAddDialog = true">+ {{ t('dictionary.addEntry') }}</el-button>
      <el-button @click="importDictionary">📥 {{ t('dictionary.importButton') }}</el-button>
      <el-button @click="exportJson">📤 {{ t('dictionary.exportJson') }}</el-button>
      <el-button @click="exportCsv">📤 {{ t('dictionary.exportCsv') }}</el-button>
    </div>

    <div class="entry-count">{{ t('dictionary.entryCount', { count: entries.length }) }}</div>

    <el-table :data="entries" stripe style="width: 100%; margin-top: 15px">
      <el-table-column prop="word" :label="t('dictionary.tableWord')" width="150" />
      <el-table-column prop="phonemes" :label="t('dictionary.tablePhonemes')" show-overflow-tooltip />
      <el-table-column :label="t('dictionary.tableActions')" width="200">
        <template #default="{ row }">
          <el-button link type="primary" size="small" @click="editEntry(row)">
            ✏️
          </el-button>
          <el-button link type="danger" size="small" @click="deleteEntry(row.word)">
            🗑️
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 添加/编辑对话框 -->
    <el-dialog v-model="showAddDialog" :title="editingWord ? '编辑' : '添加'" width="400px">
      <el-form :model="formData" label-width="80px">
        <el-form-item :label="t('dictionary.wordLabel')">
          <el-input 
            v-model="formData.word" 
            :placeholder="t('dictionary.wordPlaceholder')"
            :disabled="!!editingWord"
          />
        </el-form-item>
        <el-form-item :label="t('dictionary.phonemesLabel')">
          <el-input 
            v-model="formData.phonemes" 
            type="textarea" 
            :rows="3"
            :placeholder="t('dictionary.phonemesPlaceholder')"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddDialog = false">{{ t('app.cancel') || '取消' }}</el-button>
        <el-button type="primary" @click="saveEntry">{{ t('app.save') || '保存' }}</el-button>
      </template>
    </el-dialog>

    <!-- 导入对话框 -->
    <el-dialog v-model="showImportDialog" title="导入词典" width="400px">
      <el-upload
        ref="uploadRef"
        action="#"
        :auto-upload="false"
        :limit="1"
        accept=".csv,.json"
        @change="handleImportFileSelect"
      >
        <template #trigger>
          <el-button>{{ t('dictionary.importFile') }}</el-button>
        </template>
      </el-upload>
      <el-checkbox v-model="importOverwrite" style="margin-top: 15px">
        {{ t('dictionary.importOverwrite') }}
      </el-checkbox>
      <template #footer>
        <el-button @click="showImportDialog = false">{{ t('app.cancel') || '取消' }}</el-button>
        <el-button type="primary" @click="confirmImport">{{ t('app.import') || '导入' }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'

interface DictionaryEntry {
  word: string
  phonemes: string
}

const props = defineProps<{
  source: string
  t: any
}>()

const emit = defineEmits<{
  refresh: []
}>()

const { t } = useI18n()
const entries = ref<DictionaryEntry[]>([])
const showAddDialog = ref(false)
const showImportDialog = ref(false)
const editingWord = ref<string | null>(null)
const importOverwrite = ref(true)
const uploadRef = ref()
const importFile = ref<File | null>(null)

const formData = ref<DictionaryEntry>({
  word: '',
  phonemes: ''
})

const loadDictionary = async () => {
  try {
    const res = await fetch(`/api/dictionary/${props.source}`)
    const data = await res.json()
    if (data.success) {
      entries.value = data.entries || []
    }
  } catch (e) {
    ElMessage.error(t('dictionary.loadFailed', { error: String(e) }))
  }
}

const saveEntry = async () => {
  if (!formData.value.word.trim()) {
    ElMessage.warning('请输入单词')
    return
  }

  try {
    const res = await fetch(`/api/dictionary/${props.source}/entry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData.value)
    })
    const data = await res.json()
    if (data.success) {
      ElMessage.success(t('dictionary.addSuccess'))
      showAddDialog.value = false
      formData.value = { word: '', phonemes: '' }
      editingWord.value = null
      loadDictionary()
      emit('refresh')
    } else {
      ElMessage.error(data.error || '保存失败')
    }
  } catch (e) {
    ElMessage.error(String(e))
  }
}

const editEntry = (entry: DictionaryEntry) => {
  editingWord.value = entry.word
  formData.value = { ...entry }
  showAddDialog.value = true
}

const deleteEntry = async (word: string) => {
  try {
    const res = await fetch(`/api/dictionary/${props.source}/entry?word=${word}`, {
      method: 'DELETE'
    })
    const data = await res.json()
    if (data.success) {
      ElMessage.success(t('dictionary.deleteSuccess'))
      loadDictionary()
      emit('refresh')
    }
  } catch (e) {
    ElMessage.error(String(e))
  }
}

const importDictionary = () => {
  showImportDialog.value = true
}

const handleImportFileSelect = (file: any) => {
  importFile.value = file.raw
}

const confirmImport = async () => {
  if (!importFile.value) return

  const formData = new FormData()
  formData.append('file', importFile.value)
  formData.append('overwrite', importOverwrite.value.toString())

  try {
    const res = await fetch(`/api/dictionary/${props.source}/import`, {
      method: 'POST',
      body: formData
    })
    const data = await res.json()
    if (data.success) {
      ElMessage.success(t('dictionary.importSuccess', { added: data.added, updated: data.updated }))
      showImportDialog.value = false
      importFile.value = null
      loadDictionary()
      emit('refresh')
    } else {
      ElMessage.error(data.error || '导入失败')
    }
  } catch (e) {
    ElMessage.error(String(e))
  }
}

const exportJson = async () => {
  try {
    const res = await fetch(`/api/dictionary/${props.source}/export?format=json`)
    const data = await res.json()
    const jsonStr = JSON.stringify(data.data, null, 2)
    downloadFile(jsonStr, `${props.source}_dictionary.json`, 'application/json')
  } catch (e) {
    ElMessage.error(String(e))
  }
}

const exportCsv = async () => {
  try {
    const res = await fetch(`/api/dictionary/${props.source}/export?format=csv`)
    const csv = await res.text()
    downloadFile(csv, `${props.source}_dictionary.csv`, 'text/csv')
  } catch (e) {
    ElMessage.error(String(e))
  }
}

const downloadFile = (content: string, filename: string, mimeType: string) => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

onMounted(() => {
  loadDictionary()
})
</script>

<style scoped>
.dictionary-table-container {
  padding: 20px 0;
}

.toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.entry-count {
  color: #606266;
  font-size: 12px;
  margin-bottom: 10px;
}
</style>
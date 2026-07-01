<template>
  <div class="dictionary-manager-container">
    <el-card class="dictionary-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span class="card-title">📖 {{ t('dictionary.pageTitle') }}</span>
        </div>
      </template>

      <el-alert type="info" :closable="false" show-icon class="subtitle">
        <template #title>{{ t('dictionary.pageSubtitle') }}</template>
      </el-alert>

      <el-tabs v-model="activeTab" class="dictionary-tabs">
        <!-- SynthesizerV 词典标签页 -->
        <el-tab-pane :label="t('dictionary.sourceSynthesizerV')" name="synthesizerv">
          <div class="dictionary-content">
            <div class="hint-text">💡 {{ t('dictionary.sourceHintSynthesizerV') }}</div>
            <DictionaryTable 
              :source="'synthesizerv'"
              :t="t"
              @refresh="loadDictionary('synthesizerv')"
            />
          </div>
        </el-tab-pane>

        <!-- VOCALOID 词典标签页 -->
        <el-tab-pane :label="t('dictionary.sourceVocaloid')" name="vocaloid">
          <div class="dictionary-content">
            <div class="hint-text">💡 {{ t('dictionary.sourceHintVocaloid') }}</div>
            <DictionaryTable 
              :source="'vocaloid'"
              :t="t"
              @refresh="loadDictionary('vocaloid')"
            />
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import DictionaryTable from '../components/DictionaryTable.vue'

const { t } = useI18n()
const activeTab = ref('synthesizerv')

const loadDictionary = (source: string) => {
  console.log(`加载 ${source} 词典`)
}
</script>

<style scoped>
.dictionary-manager-container {
  width: 100%;
  padding: 20px;
}

.dictionary-card {
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

.subtitle {
  margin-bottom: 20px;
}

.hint-text {
  color: #606266;
  font-size: 12px;
  margin-bottom: 15px;
  padding: 8px 12px;
  background: #f5f5f5;
  border-radius: 4px;
  line-height: 1.5;
}

.dictionary-content {
  padding: 15px 0;
}

.dictionary-tabs :deep(.el-tabs__nav) {
  border-bottom: 2px solid #e4e7eb;
}

.dictionary-tabs :deep(.el-tabs__active-bar) {
  background-color: #409eff;
}
</style>

<template>
  <div class="app-container">
    <!-- 头部 -->
    <el-header class="app-header">
      <div class="header-content">
        <h1>🎤 GPT-SOVITS MFA 音频标注工具</h1>
        <el-tooltip content="MFA状态" placement="bottom">
          <el-tag v-if="mfaStatus" type="success" size="large">✓ MFA已安装</el-tag>
          <el-tag v-else type="danger" size="large">✗ MFA未安装</el-tag>
        </el-tooltip>
      </div>
    </el-header>

    <!-- 主体 -->
    <el-main class="app-main">
      <MFAProcessor @mfa-status-changed="checkMFAStatus" />
    </el-main>

    <!-- 页脚 -->
    <el-footer class="app-footer">
      <p>GPT-SOVITS MFA Aligner v1.0.0</p>
    </el-footer>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import MFAProcessor from './components/MFAProcessor.vue'

const mfaStatus = ref(false)

onMounted(() => {
  checkMFAStatus()
})

const checkMFAStatus = async () => {
  try {
    const res = await fetch('/api/mfa/status')
    const data = await res.json()
    mfaStatus.value = data.installed
  } catch (e) {
    console.warn('无法检查MFA状态')
    mfaStatus.value = false
  }
}
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.app-header {
  background: rgba(255, 255, 255, 0.98);
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
}

.header-content h1 {
  margin: 0;
  font-size: 24px;
  font-weight: bold;
  color: #333;
}

.app-main {
  flex: 1;
  max-width: 1000px;
  width: 100%;
  margin: 30px auto;
  padding: 0 20px;
}

.app-footer {
  background: rgba(0, 0, 0, 0.85);
  color: white;
  text-align: center;
  padding: 20px;
}

.app-footer p {
  margin: 0;
}
</style>
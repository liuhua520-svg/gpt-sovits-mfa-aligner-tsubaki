<template>
  <div class="app-container">
    <!-- 头部 -->
    <el-header class="app-header">
      <div class="header-content">
        <div class="header-left">
          <h1>歌声合成 Lab 标注工具 - SVS Lab Aligner</h1>
          <p class="subtitle">MFA自动标注 + 音高提取 + 工程文件生成</p>
        </div>
        <div class="header-right">
          <el-tooltip content="系统状态" placement="bottom">
            <el-tag v-if="systemReady" type="success" size="large">✓ 系统就绪</el-tag>
            <el-tag v-else type="danger" size="large">⚠️ 需要配置</el-tag>
          </el-tooltip>
        </div>
      </div>
    </el-header>

    <!-- 主体 -->
    <el-main class="app-main">
      <AudioProcessor @status-changed="checkSystemStatus" />
    </el-main>

    <!-- 页脚 -->
    <el-footer class="app-footer">
      <div class="footer-content">
        <p>Audio Processing Aligner v2.0.0 • Built with PyWORLD + MFA + Vue3</p>
        <p>
          <a href="https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner" target="_blank">
            📚 GitHub
          </a>
          |
          <a href="https://github.com/liuhua520-svg/gpt-sovits-mfa-aligner/issues" target="_blank">
            🐛 Issue
          </a>
        </p>
      </div>
    </el-footer>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import AudioProcessor from './components/MFAProcessor.vue'

const systemReady = ref(false)

onMounted(() => {
  checkSystemStatus()
})

const checkSystemStatus = async () => {
  try {
    const res = await fetch('/api/pipeline/status')
    const data = await res.json()
    
    if (data.success) {
      systemReady.value = data.status.mfa?.installed ?? false
    }
  } catch (e) {
    console.warn('无法检查系统状态')
    systemReady.value = false
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
  backdrop-filter: blur(10px);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  gap: 20px;
}

.header-left {
  flex: 1;
}

.header-left h1 {
  margin: 0;
  font-size: 28px;
  font-weight: bold;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  margin: 5px 0 0 0;
  color: #909399;
  font-size: 14px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.app-main {
  flex: 1;
  max-width: 1100px;
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

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
}

.footer-content p {
  margin: 8px 0;
  font-size: 12px;
}

.footer-content a {
  color: #67c23a;
  text-decoration: none;
  transition: color 0.3s;
}

.footer-content a:hover {
  color: #85ce61;
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    align-items: flex-start;
  }

  .header-left h1 {
    font-size: 20px;
  }

  .header-right {
    width: 100%;
  }
}
</style>
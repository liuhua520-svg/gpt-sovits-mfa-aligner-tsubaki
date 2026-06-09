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
/* 全局防溢出基础设置 */
* {
  box-sizing: border-box;
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #3b4175 0%, #1e1b4b 100%);
  overflow-x: hidden;
}

.app-header {
  background: rgba(255, 255, 255, 0.85);
  padding: 15px 20px;
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  
  /* 🔥 核心修复：强制解除 el-header 默认的 60px 限制，允许随文字高度自适应 */
  height: auto !important; 
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
  font-size: 26px;
  font-weight: 700;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  margin: 6px 0 0 0;
  color: #64748b;
  font-size: 13px;
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
  background: rgba(15, 12, 30, 0.95);
  color: rgba(255, 255, 255, 0.7);
  text-align: center;
  padding: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  margin-top: auto;
  
  /* 🔥 核心修复：强制解除 el-footer 默认的 60px 限制，防止底部 GitHub 链接溢出被裁切 */
  height: auto !important; 
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
  color: #a7f3d0;
  text-decoration: none;
  transition: color 0.3s;
}

.footer-content a:hover {
  color: #34d399;
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }

  .header-left h1 {
    font-size: 20px;
  }

  .header-right {
    width: 100%;
  }
}
</style>
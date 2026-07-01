<template>
  <el-config-provider :locale="elementPlusLocale">
    <div class="app-container">
      <!-- 头部 -->
      <el-header class="app-header">
        <div class="header-content">
          <div class="header-left">
            <!-- 左上角功能菜单：对齐处理 / 词典管理 / 设置 / 对话文本框 -->
            <el-dropdown trigger="click" placement="bottom-start" @command="onMenuCommand">
              <button class="menu-trigger" :aria-label="t('menu.title')" type="button">
                <el-icon :size="22"><Menu /></el-icon>
              </button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="aligner" :class="{ 'is-active-route': currentRouteName === 'aligner' }">
                    <el-icon><House /></el-icon>
                    <span>{{ t('menu.aligner') }}</span>
                  </el-dropdown-item>
                  <el-dropdown-item command="dictionary" divided :class="{ 'is-active-route': currentRouteName === 'dictionary' }">
                    <el-icon><Notebook /></el-icon>
                    <span>{{ t('menu.dictionary') }}</span>
                  </el-dropdown-item>
                  <el-dropdown-item command="dialogue" :class="{ 'is-active-route': currentRouteName === 'dialogue' }">
                    <el-icon><ChatDotRound /></el-icon>
                    <span>{{ t('menu.dialogue') }}</span>
                  </el-dropdown-item>
                  <el-dropdown-item command="settings" divided :class="{ 'is-active-route': currentRouteName === 'settings' }">
                    <el-icon><Setting /></el-icon>
                    <span>{{ t('menu.settings') }}</span>
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>

            <div class="header-titles">
              <h1>{{ t('app.title') }}</h1>
              <p class="subtitle">{{ t('app.subtitle') }}</p>
            </div>
          </div>
          <div class="header-right">
            <el-select v-model="localeModel" size="small" style="width: 120px">
              <el-option label="简体中文" value="zh-CN" />
			  <el-option label="繁體中文" value="zh-TW" />
              <el-option label="English" value="en" />
			  <el-option label="日本語" value="ja" />
              <el-option label="한국어" value="ko" />
            </el-select>
            <el-tooltip :content="t('app.systemStatus')" placement="bottom">
              <el-tag v-if="systemReady" type="success" size="large">✓ {{ t('app.ready') }}</el-tag>
              <el-tag v-else type="danger" size="large">⚠️ {{ t('app.needConfig') }}</el-tag>
            </el-tooltip>
          </div>
        </div>
      </el-header>

      <!-- 主体 -->
      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <component :is="Component" @status-changed="onSystemStatusChanged" />
        </router-view>
      </el-main>

      <!-- 页脚 -->
      <el-footer class="app-footer">
        <div class="footer-content">
          <p>{{ t('app.footer') }}</p>
          <p>
            <a href="https://github.com/liuhua520-svg/SVS-Lab-Aligner" target="_blank">
              📚 GitHub
            </a>
            |
            <a href="https://github.com/liuhua520-svg/SVS-Lab-Aligner/issues" target="_blank">
              🐛 Issue
            </a>
          </p>
        </div>
      </el-footer>
    </div>
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Menu, House, Notebook, Setting, ChatDotRound } from '@element-plus/icons-vue'
import { getElementPlusLocale, useAppLocale } from './i18n'
import { useI18n } from 'vue-i18n'

const systemReady = ref(false)
const { t, currentLocale, setLocale } = useAppLocale()
const { locale } = useI18n()
const route = useRoute()
const router = useRouter()

const currentRouteName = computed(() => route.name as string | undefined)

const onMenuCommand = (command: string) => {
  if (command !== currentRouteName.value) {
    router.push({ name: command })
  }
}

const localeModel = computed({
  get: () => currentLocale.value,
  set: (value) => setLocale(value),
})

watch(locale, (value) => {
  document.documentElement.lang = value
  localStorage.setItem('app-locale', value)
})

const elementPlusLocale = computed(() => getElementPlusLocale(currentLocale.value))

// 【修复】右上角"系统就绪"标签不再自己单独 fetch 一次 /api/pipeline/status。
// 子组件 MFAProcessor 在 onMounted 时已经会做一次完整的状态检查
// （/api/pipeline/status + /api/aligner/status），检查完之后会把结果通过
// status-changed 事件直接带上来。这里只负责消费这份数据，原因：
//   1. 避免页面一打开父子组件各自发一次几乎一样的状态请求（多余的网络/子进程开销）；
//   2. 避免"先看到底部面板更新，过一会儿右上角才更新"的不同步现象——
//      两边现在用的是同一份响应，而不是分别独立 fetch 的两份，
//      也就不会出现两边状态对不上的情况。
// 注意：该事件只由"对齐处理"页面（MFAProcessor.vue）发出；词典管理 / 设置 /
// 对话文本框三个新页面不涉及 MFA 状态，不会发出此事件，监听器闲置即可，无副作用。
const onSystemStatusChanged = (status?: { mfa?: { installed?: boolean } }) => {
  systemReady.value = status?.mfa?.installed ?? false
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
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}

.menu-trigger {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  border: 1px solid rgba(79, 70, 229, 0.18);
  background: rgba(79, 70, 229, 0.08);
  color: #4f46e5;
  cursor: pointer;
  transition: background 0.2s, transform 0.15s;
}

.menu-trigger:hover {
  background: rgba(79, 70, 229, 0.16);
}

.menu-trigger:active {
  transform: scale(0.94);
}

.header-titles {
  min-width: 0;
}

.header-titles h1 {
  margin: 0;
  font-size: 26px;
  font-weight: 700;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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

  .header-titles h1 {
    font-size: 20px;
  }

  .header-right {
    width: 100%;
  }
}
</style>


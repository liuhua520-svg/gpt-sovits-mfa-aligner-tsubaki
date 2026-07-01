import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import MFAProcessor from '../pages/MFAProcessor.vue'
import DictionaryManager from '../pages/DictionaryManager.vue'
import SettingsPage from '../pages/SettingsPage.vue'
import DialogueBatch from '../pages/DialogueBatch.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'aligner',
    component: MFAProcessor,
    meta: { label: 'aligner', icon: '🎯' }
  },
  {
    path: '/dictionary',
    name: 'dictionary',
    component: DictionaryManager,
    meta: { label: 'dictionary', icon: '📖' }
  },
  {
    path: '/settings',
    name: 'settings',
    component: SettingsPage,
    meta: { label: 'settings', icon: '⚙️' }
  },
  {
    path: '/dialogue',
    name: 'dialogue',
    component: DialogueBatch,
    meta: { label: 'dialogue', icon: '💬' }
  }
]

export const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router

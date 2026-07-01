import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'

// 延迟加载组件
const AlignerPage = () => import('./pages/AlignerPage.vue')
const DictionaryPage = () => import('./pages/DictionaryPage.vue')
const SettingsPage = () => import('./pages/SettingsPage.vue')
const DialoguePage = () => import('./pages/DialoguePage.vue')

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/aligner',
  },
  {
    path: '/aligner',
    name: 'Aligner',
    component: AlignerPage,
    meta: { title: '对齐处理' },
  },
  {
    path: '/dictionary',
    name: 'Dictionary',
    component: DictionaryPage,
    meta: { title: '词典管理' },
  },
  {
    path: '/settings',
    name: 'Settings',
    component: SettingsPage,
    meta: { title: '设置' },
  },
  {
    path: '/dialogue',
    name: 'Dialogue',
    component: DialoguePage,
    meta: { title: '对话批量处理' },
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router

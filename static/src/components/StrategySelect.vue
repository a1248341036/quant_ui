<template>
  <div class="picker" ref="root">
    <div class="picker-trigger" @click.stop="open = !open">
      <template v-if="multiple">
        <span v-if="selected.length" class="picker-summary">{{selected.length}} / {{strategies.length}} 策略</span>
        <span v-else class="picker-summary muted">{{placeholder}}</span>
        <span v-if="selected.length" class="picker-selected-chips">
          <span v-for="name in selected" :key="name" class="mini-chip" @click.stop="toggle(name)" :title="name">{{name}} ✕</span>
        </span>
      </template>
      <template v-else>
        <span class="picker-summary">{{selected || placeholder}}</span>
      </template>
      <span class="picker-arrow">▾</span>
    </div>
    <div v-if="open" class="picker-dropdown" @click.stop>
      <input v-model="search" class="picker-search" :placeholder="multiple ? '搜索策略名称/说明…' : '搜索策略…'">
      <div v-if="multiple" class="picker-actions">
        <button type="button" class="picker-action" @click="setDefaults">默认</button>
        <button type="button" class="picker-action" @click="selectAll">全选</button>
        <button type="button" class="picker-action" @click="clearAll">清空</button>
        <span class="picker-count">{{selected.length}} / {{strategies.length}}</span>
      </div>
      <div v-if="overMsg" class="warn picker-over">{{overMsg}}</div>
      <div class="picker-list">
        <template v-for="(list, g) in filtered" :key="g">
          <div class="picker-group-row">
            <span class="picker-group">{{g}}</span>
            <button type="button" class="picker-group-toggle" @click="toggleGroup(list)">{{allSelected(list) ? '本组全取消' : '本组全选'}}</button>
          </div>
          <div v-for="s in list" :key="s.name" class="picker-item" :class="{selected: multiple ? selected.includes(s.name) : selected === s.name}" @click="toggle(s.name)">
            <span class="picker-check">{{multiple ? (selected.includes(s.name) ? '☑' : '☐') : (selected === s.name ? '●' : '○')}}</span>
            <span class="picker-name">{{s.name}}</span>
            <span class="picker-desc">{{s.desc || s.factor}}</span>
          </div>
        </template>
        <div v-if="!Object.keys(filtered).length" class="picker-empty">无匹配策略</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'StrategySelect',
  props: {
    modelValue: { type: [String, Array], default: '' },
    strategies: { type: Array, default: () => [] },
    multiple: { type: Boolean, default: false },
    placeholder: { type: String, default: '选择策略' },
    max: { type: Number, default: 0 },
    defaults: { type: Array, default: () => ['低换手冷门', '反转 20 日', '低波动', '动量 20 日'] },
  },
  emits: ['update:modelValue'],
  data() { return { open: false, search: '', overMsg: '' }; },
  computed: {
    selected() {
      return this.multiple ? (Array.isArray(this.modelValue) ? this.modelValue : []) : (this.modelValue || '');
    },
    filtered() {
      const q = (this.search || '').trim().toLowerCase();
      const arr = this.strategies.filter(s =>
        !q || (s.name || '').toLowerCase().includes(q) ||
        (s.desc || '').toLowerCase().includes(q) ||
        (s.factor || '').toLowerCase().includes(q));
      const groups = {};
      for (const s of arr) {
        const g = s.group || '其他';
        (groups[g] = groups[g] || []).push(s);
      }
      return groups;
    },
  },
  methods: {
    toggle(name) {
      if (!this.multiple) { this.$emit('update:modelValue', name); this.open = false; return; }
      const sel = this.selected.slice();
      const i = sel.indexOf(name);
      if (i >= 0) { sel.splice(i, 1); }
      else if (this.max && sel.length >= this.max) {
        this.overMsg = '最多选 ' + this.max + ' 个';
        setTimeout(() => { this.overMsg = ''; }, 2000);
        return;
      } else { sel.push(name); }
      this.$emit('update:modelValue', sel);
    },
    selectAll() {
      const names = this.strategies.map(s => s.name);
      this.$emit('update:modelValue', this.max ? names.slice(0, this.max) : names);
    },
    allSelected(list) { return list.length > 0 && list.every(s => this.selected.includes(s.name)); },
    toggleGroup(list) {
      const turnOff = this.allSelected(list);
      const others = this.selected.filter(n => !list.some(s => s.name === n));
      if (turnOff) { this.$emit('update:modelValue', others); return; }
      const added = [];
      for (const s of list) {
        if (!others.includes(s.name) && (!this.max || others.length + added.length < this.max)) added.push(s.name);
      }
      this.$emit('update:modelValue', others.concat(added));
    },
    setDefaults() {
      this.$emit('update:modelValue', this.defaults.slice(0, this.max || this.defaults.length));
    },
    clearAll() { this.$emit('update:modelValue', this.multiple ? [] : ''); },
    onDocClick(e) { if (this.$refs.root && !this.$refs.root.contains(e.target)) this.open = false; },
  },
  mounted() { document.addEventListener('click', this.onDocClick); },
  beforeUnmount() { document.removeEventListener('click', this.onDocClick); },
}
</script>
